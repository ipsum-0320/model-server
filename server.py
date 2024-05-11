from flask import Flask
from predict.huadong import huadong
from predict.xibei import xibei
import loading.huadong
import loading.xibei
import config
# 这行代码用来提前加载。

# init server.
app = Flask(__name__)
app.register_blueprint(huadong)
app.register_blueprint(xibei)


@app.get("/healthz")
def alive():
    return "Alive!"


if __name__ == '__main__':
    app.run()
