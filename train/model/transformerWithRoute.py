import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from mylib.config import Configurable
from mylib.config import save

class TransformerWithRoute(pl.LightningModule, Configurable):
    input_size: save
    d_model: save
    nhead: save
    num_layers: save
    out_length: save
    opt_lr: save

    def __init__(self, input_size=5, d_model=64, nhead=8, num_layers=2, out_length=1, opt_lr=1e-3):  
        super().__init__()
        self.save_hyperparameters()

        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.out_length = out_length
        self.opt_lr = opt_lr
    
        self.linear = nn.Linear(input_size, d_model)

        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, 2)

    def forward(self, x):
        x = self.linear(x)
        output = self.transformer_encoder(x)  
        output = self.fc(output)
        return output
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.opt_lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss