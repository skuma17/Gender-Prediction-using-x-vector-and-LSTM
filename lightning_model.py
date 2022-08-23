import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio

import wavencoder
import torchmetrics
from pytorch_lightning import LightningModule
import pytorch_lightning as pl

from torchmetrics import Accuracy, ConfusionMatrix
from torchmetrics import F1Score, AUROC
from torchmetrics import MeanSquaredError  as MSE
from torchmetrics import MeanAbsoluteError as MAE



from models import Wav2VecLSTM_Base
from models import SpeechBrainLSTM


import pandas as pd
import wavencoder

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

class LightningModel(pl.LightningModule):
    def __init__(self, HPARAMS):
        super(LightningModel, self).__init__()
        # HPARAMS
        self.save_hyperparameters()
        self.model = SpeechBrainLSTM(HPARAMS['model_hidden_size'])
        #self.model = Wav2VecLSTM_Base(HPARAMS['model_hidden_size'])

        self.ConfutionMatrix_BinaryClass_criterion = ConfusionMatrix(num_classes=2, threshold=0.5, Multilabel=False )
        self.F1_criterion = F1Score(number_classes=2,
        average="micro")
        

        self.classification_criterion = MSE()
        self.regression_criterion = MSE()
        self.mae_criterion = MAE()
        self.rmse_criterion = RMSELoss()
        self.accuracy = Accuracy()

        self.alpha = HPARAMS['model_alpha']
        self.beta = HPARAMS['model_beta']
        self.gamma = HPARAMS['model_gamma']

        self.lr = HPARAMS['training_lr']

        
        self.csv_path = HPARAMS['speaker_csv_path']
        self.df = pd.read_csv(self.csv_path, sep=' ')
        
        self.a_mean = self.df['Age'].mean()
        self.a_std = self.df['Age'].std()
        
        print(f"Model Details: #Params = {self.count_total_parameters()}\t#Trainable Params = {self.count_trainable_parameters()}")

    def count_total_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
  
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        x, y_g = batch
        y_hat_g = self(x)

        y_g = y_g.view(-1).float()
        y_hat_g = y_hat_g.view(-1).float()

        gender_loss = self.classification_criterion(y_hat_g, y_g)
        loss = self.gamma * gender_loss

        gender_acc = self.accuracy((y_hat_g>0.5).long(), y_g.long())
        g_F1Score = self.F1_criterion((y_hat_g>0.5).long(), y_g.long())
        
        #y_hat_nl = y_hat_nl.argmax(axis=1)
        #g_confmatrix = self.ConfutionMatrix_BinaryClass_criterion.update(y_hat_g, y_g)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        return {'loss':loss, 
                'train_gender_acc':gender_acc,
                'train_g_F1score':g_F1Score,
#                'train_g_Confmatrix':g_confmatrix,
                 }
    
    def training_epoch_end(self, outputs):
        n_batch = len(outputs)
        loss = torch.tensor([x['loss'] for x in outputs]).mean()
                
        gender_acc = torch.tensor([x['train_gender_acc'] for x in outputs]).mean()        
        g_F1Score = torch.tensor([x['train_g_F1score'] for x in outputs]).mean()
        #g_confmatrix = self.ConfutionMatrix_BinaryClass_criterion.compute()
        
        self.log('epoch_loss' , loss, prog_bar=True, sync_dist=True)
        self.log('gender_accuracy',gender_acc, prog_bar=True, sync_dist=True)
        self.log('gender F1Score',g_F1Score, prog_bar=True, sync_dist=True)
        
    def validation_step(self, batch, batch_idx):
        x, y_g = batch       
        y_hat_g = self(x)

        y_g = y_g.view(-1).float()
        y_hat_g = y_hat_g.view(-1).float()

        gender_loss = self.classification_criterion(y_hat_g, y_g)
        loss = self.gamma * gender_loss
        
        gender_acc = self.accuracy((y_hat_g>0.5).long(), y_g.long())

        return {'val_loss':loss, 
                'val_gender_acc':gender_acc}


    def validation_epoch_end(self, outputs):
        n_batch = len(outputs)
        val_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        
        gender_acc = torch.tensor([x['val_gender_acc'] for x in outputs]).mean()

        self.log('v_loss' , val_loss, prog_bar=True, sync_dist=True)
        self.log('v_g_acc',gender_acc, prog_bar=True, sync_dist=True)
        
        
    def test_step(self, batch, batch_idx):       
        x, y_g = batch
        y_hat_g = self(x)

        y_g = y_g.view(-1).float()
        y_hat_g = y_hat_g.view(-1).float()

        gender_acc = self.accuracy((y_hat_g>0.5).long(), y_g.long())

        return {        
                'test_gender_acc':gender_acc}


    def test_epoch_end(self, outputs):
        n_batch = len(outputs)

        gender_acc = torch.tensor([x['test_gender_acc'] for x in outputs]).mean()

        pbar = {'test_gender_acc':gender_acc.item()} 

        self.logger.log_hyperparams(pbar)
        self.log_dict(pbar)
        

