from abc import ABC
import os
import yaml
from typing import get_type_hints

class save:
    pass

class Configurable(ABC):
    def get_config(self):
        """save アノテーションが付いている項目だけを dict 形式で返す"""
        config_items = {}
        type_hints = get_type_hints(type(self))  # 型ヒントを取得
        for attr_name, attr_value in self.__dict__.items():
            if attr_name in type_hints and type_hints[attr_name] == save:
                config_items[attr_name] = attr_value
        config_items["_name"] = type(self).__name__
        return config_items

    @classmethod
    def from_config(cls, config):
        """configを元にインスタンスを生成する"""
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
        print(self.config)
        to_yaml(self.config, path)
    

def to_yaml(config, path) -> None:
    with open(os.path.join(path, "config.yaml") , 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def from_yaml(path) -> dict:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data