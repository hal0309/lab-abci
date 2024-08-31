import os
import sys
import matplotlib.pyplot as plt
import pickle
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

import model._models as models
from datamodule.datamodule import MyDataModule
from datamodule.datamodule import MyDataModuleWithRoute
import mylib.route as m_route
import mylib.utils as ut
import mylib.config as conf


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DF_PATH = os.path.join(ROOT_PATH, "data", "df_test.pickle")
OUTPUT_DIR = os.path.join(ROOT_PATH, "out")
CONFIG_DIR = os.path.join(ROOT_PATH, "config")




def main():
    # 乱数シードの固定
    ut.fix_seeds(0)

    # 引数の取得
    config_name, max_epochs = check_args(sys.argv)
    
    df = pickle.load(open(DF_PATH, "rb"))
    
    config_path = os.path.join(CONFIG_DIR, config_name)
    config = conf.from_yaml(config_path)

    route_gen = m_route.DistanceRotateRouteGeneraterV1.from_config(config["route"])

    config_dm = config["dm"]
    # dm = MyDataModule(n_of_route=config_dm["n_of_route"], batch_size=config_dm["batch_size"], route_gen=route_gen, df=df)
    dm = MyDataModuleWithRoute(n_of_route=config_dm["n_of_route"], batch_size=config_dm["batch_size"], route_gen=route_gen, df=df)


    model = models.get_model(config["model"])

    fname = config["fname"]

    # ディレクトリの作成
    log_fname = f"{ut.get_datetime()}-{fname}"
    output_dir = os.path.join(OUTPUT_DIR, "lightning_logs")
    checkpoint_dir = os.path.join(output_dir, log_fname, "cp")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # configの保存
    config_builder = conf.ConfigrationBuilder()
    config_builder.add("route", route_gen.get_config())
    config_builder.add("dm", dm.get_config())
    config_builder.add("model", model.get_config())
    config_builder.to_yaml(output_dir)

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
    
    trainer.fit(model, dm)    



def check_args(argv):
    if len(sys.argv) == 3:
        config_name = sys.argv[1]
        max_epochs = int(sys.argv[2])
    elif len(sys.argv) == 2:
        config_name = sys.argv[1]
        max_epochs = 10000
    else:
        raise ValueError("Invalid arguments length (train.py config_name [max_epochs])")
    
    return config_name, max_epochs



if __name__ == "__main__":
    main()