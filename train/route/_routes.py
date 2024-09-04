from route.distanceRotateRouteGeneraterV1 import DistanceRotateRouteGeneraterV1


DICT = {
        "DistanceRotateRouteGeneraterV1": DistanceRotateRouteGeneraterV1
    }


def get_route_generator(config):
    try:
        return DICT[config["_name"]].from_config(config)
    except KeyError:
        raise ValueError(f"model_name: {config['_name']} is not supported.")