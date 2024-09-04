from datamodule.myDatamodule import MyDataModule


DICT = {
        "MyDataModule": MyDataModule
    }


def get_dm(config):
    try:
        return DICT[config["_name"]].from_config(config)
    except KeyError:
        raise ValueError(f"model_name: {config['_name']} is not supported.")
