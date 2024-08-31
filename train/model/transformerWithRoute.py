import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from mylib.config import Configurable
from mylib.config import save

class TransformerWithRoute(pl.LightningModule, Configurable):
    d_model: save
    nhead: save
    num_layers: save
    out_length: save

    def __init__(self, d_model=64, nhead=8, num_layers=2, out_length=1):  
        super().__init__()
        self.save_hyperparameters()

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.out_length = out_length
        
        input_size = 5

        # Ensure embed_dim (d_model) is divisible by nhead
        self.linear = nn.Linear(input_size, d_model)

        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 2)  # Output coordinates

    def forward(self, x):
        # print("x---------")
        # print(x)
        x = self.linear(x)
        # print("x-linear--------")
        # print(x)
        output = self.transformer_encoder(x)  
        # print("output1---------")
        # print(output)
        output = self.fc(output)  # Get the output for the last time step
        # print("output2---------")
        # print(output)
        return output
    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

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