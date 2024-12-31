from model.lstmByPL import LSTMByPL
from model.transformerByPL import TransformerByPL
from model.transformerWithRoute import TransformerWithRoute
from model.transformerDirect import TransformerDirect
from model.transformerDecoder import TransformerWithRouteDecoder
from model.transformerDecoderAR import TransformerWithRouteDecoderAR


DICT = {
        "LSTMByPL": LSTMByPL,
        "TransformerByPL": TransformerByPL,
        "TransformerWithRoute": TransformerWithRoute,
        "TransformerDirect": TransformerDirect,
        "TransformerWithRouteDecoder": TransformerWithRouteDecoder, # Teacher Forcing
        "TransformerWithRouteDecoderAR": TransformerWithRouteDecoderAR # Auto Regressive(自己回帰)
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