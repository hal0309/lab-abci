from model.lstmByPL import LSTMByPL
from model.transformerByPL import TransformerByPL
from model.transformerWithRoute import TransformerWithRoute


MODELS = {
        "LSTMByPL": LSTMByPL,
        "TransformerByPL": TransformerByPL,
        "TransformerWithRoute": TransformerWithRoute,
    }


def get_model(config):
    try:
        return MODELS[config["_name"]].from_config(config)
    except KeyError:
        raise ValueError(f"model_name: {config['_name']} is not supported.")
