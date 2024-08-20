import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from mylib.config import Configurable
from mylib.config import save


class TransformerByPL(pl.LightningModule, Configurable):
    d_model: save
    nhead: save
    num_layers: save
    use_distance: save

    def __init__(self, d_model=64, nhead=8, num_layers=2, use_distance=True):  
        super().__init__()
        self.save_hyperparameters()

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.use_distance = use_distance
        
        input_size = 3
        if use_distance:
            input_size += 1

        # Ensure embed_dim (d_model) is divisible by nhead
        self.linear = nn.Linear(input_size, d_model)

        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 2)  # Output coordinates

    def forward(self, x):
        if not self.use_distance:
            x = x[:, :, :3]  # Remove distance feature

        x = self.linear(x)
        
        # Add positional encoding for sequential information (if needed)
        output = self.transformer_encoder(x)  
        output = self. fc(output[:, -20, :])  # Get the output for the last time step
        return output
    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y[:,-20,:])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y[:,-20,:])
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y[:,-20,:])
        self.log("val_loss", loss, prog_bar=True)
        return loss

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
