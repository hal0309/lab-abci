from dataset.routeDataset import RouteDataset
from dataset.routeDatasetWithRoute import RouteDatasetWithRoute


DICT = {
        "RouteDataset": RouteDataset,
        "RouteDatasetWithRoute": RouteDatasetWithRoute
    }


def get_dataset(config):
    try:
        return DICT[config["_name"]].from_config(config)
    except KeyError:
        raise ValueError(f"model_name: {config['_name']} is not supported.")
