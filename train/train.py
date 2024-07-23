import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import datetime

import model.models as models
import mylib.route as m_route
import mylib.utils as ut
import mylib.config as conf


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DF_PATH = os.path.join(ROOT_PATH, "data", "df_test.pickle")
OUTPUT_DIR = os.path.join(ROOT_PATH, "out")

PATH_LENGTH = 40
NUMBER_OF_ROUTE = 10000


def main():
    df = pickle.load(open(DF_PATH, "rb"))
    
    batch = 64
    max_epochs = 20000
    
    n_of_route = NUMBER_OF_ROUTE

    data_name = "tf_nodist_angle_in40_out1"
    train(df, batch=batch, max_epochs=max_epochs, use_distance=False, data_name=data_name, n_of_route=n_of_route)
    
    # data_name = "tf_angle_in40_out1_distanceallzero"
    # train(df, batch=batch, max_epochs=max_epochs, use_distance=True, data_name=data_name, n_of_route=n_of_route)

    # dir_name = "2024-06-27-00-47-tf_nodist_angle_in40_out1-b64-n10000"
    # re_train(df, batch=batch, max_epochs=max_epochs, dir_name=dir_name)


def train(df, batch, max_epochs, use_distance, data_name, n_of_route):
    log_fname = f"{ut.get_datetime()}-use_dist-{use_distance}-{data_name}-b{batch}-n{n_of_route}"

    output_dir = os.path.join(OUTPUT_DIR, "lightning_logs")
    checkpoint_dir = os.path.join(output_dir, log_fname, "cp")
    os.makedirs(checkpoint_dir, exist_ok=True)


    

    ut.fix_seeds(0)

    x_max = df["x"].max()
    y_max = df["y"].max()

    # route_gen = m_route.DistanceRouteGenerater(x_max, y_max, PATH_LENGTH)
    # route_gen1 = m_route.DistanceRotateRouteGeneraterV1(x_max, y_max, PATH_LENGTH, dist_min=1, dist_max=5, angle_min=0, angle_max=90)
    # conf.to_yaml(route_gen1.get_config(), os.path.dirname(checkpoint_dir))
    config = os.path.join(ROOT_PATH, "data","config.yaml")
    # config = os.path.join(os.path.dirname(checkpoint_dir), "config.yaml")
    config = conf.from_yaml(config)
    route_gen = m_route.DistanceRotateRouteGeneraterV1.from_config(config)
    # conf.to_yaml(route_gen.get_config(), os.path.dirname(checkpoint_dir))
    conf.ConfigrationBuilder().add("route", config).to_yaml(os.path.dirname(checkpoint_dir))
    dm = MyDataModule(df, NUMBER_OF_ROUTE, route_gen, batch_size=batch)

    

    loss_checkpoint = ModelCheckpoint(
        filename=f"best_loss_fold",
        dirpath=checkpoint_dir,
        monitor="val_loss",
        save_last=True,
        save_top_k=1,
        mode="min"
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=[pl_loggers.TensorBoardLogger(output_dir, name=log_fname)],
        callbacks=[loss_checkpoint]
        )
    print(loss_checkpoint.dirpath)
    # model = LSTMByPL(hidden_size=200, use_distance=use_distance)
    model = models.TransformerByPL(use_distance=use_distance)
    # model = TransformerByPL(use_distance=use_distance)
    trainer.fit(model, dm)


# def re_train(df, batch, max_epochs, dir_name):

#     fix_seeds(0)

#     x_max = df["x"].max()
#     y_max = df["y"].max()

#     # route_gen = m_route.DistanceRouteGenerater(x_max, y_max, PATH_LENGTH)
#     route_gen = m_route.DistanceRotateRouteGenerater(x_max, y_max, PATH_LENGTH, dist_min=1, dist_max=5, angle_min=0, angle_max=90)
#     dm = MyDataModule(df, NUMBER_OF_ROUTE, route_gen, batch_size=batch)

#     loss_checkpoint = ModelCheckpoint(
#         filename=f"best_loss_fold",
#         dirpath=dir_name,
#         monitor="val_loss",
#         save_last=True,
#         save_top_k=1,
#         mode="min"
#     )

#     checkpoint_path = f"./lightning_logs/{dir_name}/cp/last.ckpt"
    
#     trainer = pl.Trainer(
#         max_epochs=max_epochs,
#         logger=[pl_loggers.TensorBoardLogger("lightning_logs", name=dir_name)],
#         callbacks=[loss_checkpoint]
#         )

#     print(loss_checkpoint.dirpath)
#     model = TransformerByPL.load_from_checkpoint(checkpoint_path)
#     trainer.fit(model, dm, ckpt_path=checkpoint_path)






class MyDataModule(pl.LightningDataModule):
    def __init__(self, df, n_of_route, route_gen, batch_size=1):
        super().__init__()
        self.batch_size = batch_size
        test_size = int(n_of_route * 0.2)
        train_size = int((n_of_route - test_size) * 0.8)
        val_size = n_of_route - train_size - test_size

        self.train_dataset = RouteDataset(df, train_size, route_gen)
        self.val_dataset = RouteDataset(df, val_size, route_gen)
        self.test_dataset = RouteDataset(df, test_size, route_gen)
        

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=7, persistent_workers=True)

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=7, persistent_workers=True)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=7, persistent_workers=True)
    

def getMF(df, X, Y):
    MF = []
    for x, y in zip(X, Y):
        try:
            # MF.append(df[(df["x"] == x) & (df["y"] == y)]["MF"].values[0])
            d = df[(df["x"] == x) & (df["y"] == y)]
            MF.append([d["MF_X"].values[0], d["MF_Y"].values[0], d["MF_Z"].values[0]])
        except:
            print(f"Error: {x}, {y}")
            MF.append(-1)
    return MF

class RouteDataset(data.Dataset):
    def __init__(self, df, n_of_route, route_gen) -> None:
        self.df = df
        self.size = n_of_route
        self.route_gen = route_gen
        self.route = route_gen.get_route_list(n_of_route)

    def __len__(self) -> int:
        return len(self.route)

    # -> tuple[torch.Tensor, torch.Tensor]
    def __getitem__(self, idx):
        # print("getitem")
        X = self.route[idx][0]
        Y = self.route[idx][1]

        MF = getMF(self.df, X, Y)
        
        X = [float(x * 0.1) for x in X]
        Y = [float(y * 0.1) for y in Y]
        XY = np.column_stack((X, Y))

        # distance = [np.sqrt((X[i] - X[i + 1])**2 + (Y[i] - Y[i + 1])**2) for i in range(len(X) - 1)]        
        # distance.insert(0, 0)

        distance = [0 for i in range(len(X))]    

        mf_and_d = torch.cat([torch.Tensor(MF), torch.Tensor(distance).unsqueeze(1)], dim=1)
        
        return mf_and_d, torch.Tensor(XY)
    

if __name__ == "__main__":
    main()