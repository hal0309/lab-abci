from dataset.routeDataset import RouteDataset
from dataset.routeDatasetWithRoute import RouteDatasetWithRoute
from dataset.routeDatasetWithRouteIndex import RouteDatasetWithRouteIndex
from dataset.routeDatasetWithRouteDivideIndex import RouteDatasetWithRouteDivideIndex
from dataset.routeDatasetWithRouteDiff import RouteDatasetWithRouteDiff
from dataset.routeDatasetWithZeros import RouteDatasetWithZeros
from dataset.routeDatasetWithZerosIndex import RouteDatasetWithZerosIndex
from dataset.routeDatasetWithDistance import RouteDatasetWithDistance
from dataset.routeDatasetWithDistanceIndex import RouteDatasetWithDistanceIndex
from dataset.testDataset import TestDataset


DICT = {
        "RouteDataset": RouteDataset,
        "RouteDatasetWithRoute": RouteDatasetWithRoute,
        "RouteDatasetWithRouteIndex": RouteDatasetWithRouteIndex,
        "RouteDatasetWithRouteDivideIndex": RouteDatasetWithRouteDivideIndex,
        "RouteDatasetWithRouteDiff": RouteDatasetWithRouteDiff,
        "RouteDatasetWithDistance": RouteDatasetWithDistance,
        "RouteDatasetWithDistanceIndex": RouteDatasetWithDistanceIndex,
        "RouteDatasetWithZeros": RouteDatasetWithZeros,
        "RouteDatasetWithZeros": RouteDatasetWithZeros,
        "RouteDatasetWithZerosIndex": RouteDatasetWithZerosIndex,
        "TestDataset": TestDataset,
    }


def get_dataset(config):
    try:
        return DICT[config["_name"]].from_config(config)
    except KeyError:
        raise ValueError(f"model_name: {config['_name']} is not supported.")
