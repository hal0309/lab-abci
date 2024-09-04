import numpy as np
import torch
import torch.utils.data as data
import pytorch_lightning as pl

from mylib.config import Configurable
from mylib.config import save

class MyDataModule(pl.LightningDataModule, Configurable):
    batch_size: save

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        
    def setDataset(self, dataset):
        test_size = int(len(dataset) * 0.2)
        train_size = int((len(dataset) - test_size) * 0.8)
        val_size = len(dataset) - train_size - test_size
        self.train_dataset, self.val_dataset, self.test_dataset = data.random_split(dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=7, persistent_workers=True)

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=7, persistent_workers=True)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=7, persistent_workers=True)
