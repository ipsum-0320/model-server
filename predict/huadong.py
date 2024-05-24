from flask import request, jsonify, Blueprint
from loading.huadong import huadong_models
from predict.common import predict_common, bounce

# 用来控制测试环境的参数。
dry_run = False

huadong = Blueprint('huadong', __name__)


@huadong.post("/predict/huadong/hangzhou")
def predict_huadong_hangzhou():
    # 获取参数，可以通过 data['key'] 来获取 POST 中的 json/files 参数。
    files = request.files
    model = huadong_models['hangzhou']
    res = predict_common(files=files, model=model, dry_run=dry_run)
    res = bounce(res)
    return jsonify(res)


@huadong.post("/predict/huadong/ningbo")
def predict_huadong_ningbo():
    # 获取参数，可以通过 data['key'] 来获取 POST 中的 json/files 参数。
    files = request.files
    model = huadong_models['ningbo']
    res = predict_common(files=files, model=model, dry_run=dry_run)
    res = bounce(res)
    return jsonify(res)
