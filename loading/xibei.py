# 引入西北下各个边缘站点的模型。

from models import TimesNet
import torch
from config import config
import os

zone = 'xibei'
site_list = ['xian', 'lanzhou']

xibei_models = {}

for site in site_list:
    model = TimesNet.Model(config).float().to(torch.device('cpu'))
    print('loading model')
    path = os.path.join(os.getcwd(), './checkpoints/{}/{}-checkpoint.pth'.format(zone, site))
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    for key in list(checkpoint.keys()):
        if 'module.' in key:
            checkpoint[key.replace('module.', '')] = checkpoint[key]
            del checkpoint[key]
    model.load_state_dict(checkpoint)
    print('loading {}-{}-model successfully'.format(zone, site))
    model.eval()
    xibei_models[site] = model



