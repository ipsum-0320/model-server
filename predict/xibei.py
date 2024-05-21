from flask import request, jsonify, Blueprint
from loading.xibei import xibei_models
from predict.common import predict_common, bounce

# 用来控制测试环境的参数。
dry_run = True

xibei = Blueprint('xibei', __name__)


@xibei.post("/predict/xibei/xian")
def predict_xibei_xian():
    # 获取参数，可以通过 data['key'] 来获取 POST 中的 json/files 参数。
    files = request.files
    model = xibei_models['xian']
    res = predict_common(files=files, model=model, dry_run=dry_run)
    res = bounce(res)
    return jsonify(res)


@xibei.post("/predict/xibei/lanzhou")
def predict_xibei_lanzhou():
    # 获取参数，可以通过 data['key'] 来获取 POST 中的 json/files 参数。
    files = request.files
    model = xibei_models['lanzhou']
    res = predict_common(files=files, model=model, dry_run=dry_run)
    res = bounce(res)
    return jsonify(res)
