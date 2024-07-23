from abc import ABC
import os
import yaml

class Configurable(ABC):
    def get_config(self):
        """引数をdict形式で返す(private変数を除外)"""
        config = {**self.__dict__, "_name": type(self).__name__}
        prefix = f"_{type(self).__name__}"
        return {k: v for k, v in config.items() if not k.startswith(prefix)}

    @classmethod
    def from_config(cls, config):
        """Instantiate the class from a configuration dictionary."""
        config_without_name = config.copy()
        name = config_without_name.pop("_name")

        if name != cls.__name__:
            raise ValueError(f"Invalid name '{name}' for class '{cls.__name__}'")

        return cls(**config_without_name)
    

class ConfigrationBuilder:
    def __init__(self):
        self.config = {}

    def add(self, key, config):
        self.config[key] = config
        return self

    def build(self):
        return self.config
    
    def to_yaml(self, path):
        to_yaml(self.config, path)
    


def to_yaml(config, path) -> None:
    with open(os.path.join(path, "config.yaml") , 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def from_yaml(path) -> dict:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data