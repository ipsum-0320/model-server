from utils import yaml
from types import SimpleNamespace

config = yaml.load_yaml("model.config.yaml")
config = SimpleNamespace(**config)
