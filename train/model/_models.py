from model.lstmByPL import LSTMByPL
from model.transformerByPL import TransformerByPL
from model.transformerWithRoute import TransformerWithRoute


DICT = {
        "LSTMByPL": LSTMByPL,
        "TransformerByPL": TransformerByPL,
        "TransformerWithRoute": TransformerWithRoute,
    }


def get_model(config):
    try:
        return DICT[config["_name"]].from_config(config)
    except KeyError:
        raise ValueError(f"model_name: {config['_name']} is not supported.")

def get_model_with_checkpoint(config, cp_path):
    try:
        return DICT[config["_name"]].load_from_checkpoint(cp_path)
    except KeyError:
        raise ValueError(f"model_name: {config['_name']} is not supported.")