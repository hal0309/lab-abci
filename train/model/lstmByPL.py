import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from mylib.config import Configurable
from mylib.config import save

class LSTMByPL(pl.LightningModule):
    def __init__(self, hidden_size, use_distance):
        super().__init__()
        self.save_hyperparameters()
        self.use_distance = use_distance
        input_size = 3
        if use_distance:
            input_size += 1
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 2)

    
    def forward(self, x):        
        if self.use_distance == False:
            # x[3]を削除
            x = x[:, :, :3]

        x, (h, c) = self.lstm(x)
        x = self.fc(x[:,-1,:])
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # if(batch_idx == 0):
        #     print("idx0")
        #     print(x)
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y[:,-1,:])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y[:,-1,:])
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y[:,-1,:])
        self.log("val_loss", loss, prog_bar=True)
        return loss
