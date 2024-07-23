import os
import sys
import matplotlib.pyplot as plt
import pickle
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

import model.models as models
from model.datamodule import MyDataModule
import mylib.route as m_route
import mylib.utils as ut
import mylib.config as conf


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DF_PATH = os.path.join(ROOT_PATH, "data", "df_test.pickle")
OUTPUT_DIR = os.path.join(ROOT_PATH, "out")
CONFIG_DIR = os.path.join(ROOT_PATH, "config")


def main():
    ut.fix_seeds(0)

    max_epochs = 10000

    if len(sys.argv) == 3:
        config_name = sys.argv[1]
        max_epochs = int(sys.argv[2])
    elif len(sys.argv) == 2:
        config_name = sys.argv[1]
    else:
        raise ValueError("Invalid arguments length (train.py config_name [max_epochs])")
    

    df = pickle.load(open(DF_PATH, "rb"))
    
    config_path = os.path.join(CONFIG_DIR, config_name)
    config = conf.from_yaml(config_path)

    config_route = config["route"]
    route_gen = m_route.DistanceRotateRouteGeneraterV1.from_config(config_route)

    config_dm = config["dm"]
    dm = MyDataModule(n_of_route=config_dm["n_of_route"], batch_size=config_dm["batch_size"], route_gen=route_gen, df=df)

    config_model = config["model"]
    model = models.TransformerByPL.from_config(config_model)

    fname = config["fname"]
    train(route_gen=route_gen, dm=dm, model=model, max_epochs=max_epochs, fname=fname)


def train(route_gen, dm, model, max_epochs, fname):
    log_fname = f"{ut.get_datetime()}-{fname}"

    output_dir = os.path.join(OUTPUT_DIR, "lightning_logs")
    checkpoint_dir = os.path.join(output_dir, log_fname, "cp")
    os.makedirs(checkpoint_dir, exist_ok=True)

    config_route = route_gen.get_config()
    config_dm = dm.get_config()
    config_model = model.get_config()
    conf.ConfigrationBuilder().add("route", config_route).add("dm", config_dm).add("model", config_model).add("fname", fname).to_yaml(os.path.dirname(checkpoint_dir))

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


if __name__ == "__main__":
    main()