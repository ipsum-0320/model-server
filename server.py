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
dry_run = True

@app.post("/predict")
def predict():
  # 获取参数，可以通过 data['key'] 来获取 POST 中的 json/files 参数。
  files = request.files
  if dry_run:
    batch_x, batch_x_mark, true_x = reader.convert(files['source'], config, dry_run=dry_run, start=1080)
    true_x = torch.tensor(true_x).unsqueeze(0)
    true_x = true_x.float().to(torch.device('cpu'))
  else:
    batch_x, batch_x_mark = reader.convert(files['source'], config, dry_run=dry_run, start=0)
  
  batch_x = torch.tensor(batch_x).unsqueeze(0)
  batch_x_mark = torch.tensor(batch_x_mark).unsqueeze(0)
  

  with torch.no_grad():
    batch_x = batch_x.float().to(torch.device('cpu'))
    batch_x_mark = batch_x_mark.float().to(torch.device('cpu'))

    # decoder input
    dec_inp = torch.zeros_like(batch_x[:, -config.pred_len:, :]).float()
    # encoder - decoder
    if config.use_amp:
        with torch.cuda.amp.autocast():
            if config.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_x_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_x_mark)
    else:
        if config.output_attention:
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_x_mark)[0]

        else:
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_x_mark)

    f_dim = -1 if config.features == 'MS' else 0
    outputs = outputs[:, -config.pred_len:, :]
    outputs = outputs.detach().cpu().numpy()
    outputs = outputs[:, :, f_dim:]
    pred = reader.flatten_list(outputs.tolist())

    if not dry_run:
      res = {
        'length': len(pred),
        'pred': pred
      }
      return jsonify(res)
    else:
      true_x = true_x.to(torch.device('cpu'))
      true_x = true_x.detach().cpu().numpy()
      true_x = true_x[:, :, f_dim:]
      true = reader.flatten_list(true_x.tolist())

      pred_np = np.array(outputs)
      true_np = np.array(true_x)
      pred_np = pred_np.reshape(-1, pred_np.shape[-2], pred_np.shape[-1])
      true_np = true_np.reshape(-1, true_np.shape[-2], true_np.shape[-1])
      mae, mse, rmse, mape, mspe = metric(pred_np, true_np)

      res = {
        'length': len(pred) if len(pred) == len(true) else -1,
        'pred': pred,
        'true': true,
        'mae': mae.tolist(),
        'mse': mse.tolist(),
        'rmse': rmse.tolist(),
        'mape': mape.tolist(),
        'mspe': mspe.tolist()
      }
    # 返回 json 结果。
    return jsonify(res)


@app.get("/alive")
def alive():
  return "Alive!"

if __name__ == '__main__':
  app.run()