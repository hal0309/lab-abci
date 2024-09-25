import os
import sys
import pickle
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

import model._models as models
import datamodule._datamodules as datamodules
import dataset._datasets as datasets
import route._routes as routes
import mylib.utils as ut
import mylib.config as conf

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DF_DIR = os.path.join(ROOT_PATH, "data")
OUTPUT_DIR = os.path.join(ROOT_PATH, "out")
CONFIG_DIR = os.path.join(ROOT_PATH, "config")

def main():
    # 乱数シードの固定
    ut.fix_seeds(0)

    # 引数の取得
    config_name, max_epochs = check_args(sys.argv)
    
    # configの読込
    config_path = os.path.join(CONFIG_DIR, config_name)
    config = conf.from_yaml(config_path)

    # データフレームの読込
    df_path = os.path.join(DF_DIR, config["df_name"])
    df = pickle.load(open(df_path, "rb"))

    # 各種初期化
    model = models.get_model(config["model"])
    route_gen = routes.get_route_generator(config["route"])
    dataset = datasets.get_dataset(config["dataset"])
    dm = datamodules.get_dm(config["dm"])
    
    dataset.set_route(df, route_gen)
    dm.setDataset(dataset)

    # ディレクトリの作成
    fname = config["fname"]
    log_fname = f"{ut.get_datetime()}-{fname}"
    output_dir = os.path.join(OUTPUT_DIR, "lightning_logs")
    checkpoint_dir = os.path.join(output_dir, log_fname, "cp")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # configの保存
    config_builder = conf.ConfigrationBuilder()
    config_builder.add("fname", fname)
    config_builder.add("route", route_gen.get_config())
    config_builder.add("dm", dm.get_config())
    config_builder.add("dataset", dataset.get_config())
    config_builder.add("model", model.get_config())
    config_builder.to_yaml(os.path.dirname(checkpoint_dir))

    # callback等設定
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
    
    # 学習
    trainer.fit(model, dm)    



def check_args(argv):
    if len(sys.argv) == 3: # train.py config_name max_epochs
        config_name = sys.argv[1]
        max_epochs = int(sys.argv[2])
    elif len(sys.argv) == 2: # train.py config_name
        config_name = sys.argv[1]
        max_epochs = 10000
    else:
        raise ValueError("Invalid arguments length (train.py config_name [max_epochs])")
    
    return config_name, max_epochs



if __name__ == "__main__":
    main()