from dataset.routeDataset import RouteDataset
from dataset.routeDatasetWithRoute import RouteDatasetWithRoute
from dataset.routeDatasetWithRouteDiff import RouteDatasetWithRouteDiff
from dataset.routeDatasetWithZeros import RouteDatasetWithZeros
from dataset.routeDatasetWithDistance import RouteDatasetWithDistance
from dataset.testDataset import TestDataset


DICT = {
        "RouteDataset": RouteDataset,
        "RouteDatasetWithRoute": RouteDatasetWithRoute,
        "RouteDatasetWithRouteDiff": RouteDatasetWithRouteDiff,
        "RouteDatasetWithDistance": RouteDatasetWithDistance,
        "RouteDatasetWithZeros": RouteDatasetWithZeros,
        "TestDataset": TestDataset,
    }


def get_dataset(config):
    try:
        return DICT[config["_name"]].from_config(config)
    except KeyError:
        raise ValueError(f"model_name: {config['_name']} is not supported.")
