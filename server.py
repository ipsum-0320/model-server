from flask import Flask, request, jsonify
from models import TimesNet
from utils import yaml
from types import SimpleNamespace
import torch
from utils import reader
import copy
from utils.metrics import metric
import numpy as np


# init server.
app = Flask(__name__)

config = yaml.load_yaml("model.config.yaml")
config = SimpleNamespace(**config)

# load model
model = TimesNet.Model(config).float().to(torch.device('cpu'))
print('loading model')
checkpoint = torch.load('./checkpoints/checkpoint.pth', map_location=torch.device('cpu'))
for key in list(checkpoint.keys()):
  if 'module.' in key:
    checkpoint[key.replace('module.', '')] = checkpoint[key]
    del checkpoint[key]
model.load_state_dict(checkpoint)
print('loading model successfully')
model.eval()

# 用来控制测试环境的参数。
dry_run = False

# TODO: 
# 1. 抽离 predict，按 Zone_id 分类。
# 2. 提前加载 model，避免每次请求都加载。
# 3. 将公共的 predict 部分提取到 common 中，优化代码结构，提高代码复用性。
# 4. 将请求中的 request file 使用 processor 强化一遍。

@app.get("/healthz")
def alive():
  return "Alive!"

if __name__ == '__main__':
  app.run()