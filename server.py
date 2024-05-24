from flask import Flask
from predict.huadong import huadong
import loading.huadong
import config
# 这行代码用来提前加载。

# init server.
app = Flask(__name__)
app.register_blueprint(huadong)


@app.get("/healthz")
def alive():
    return "Alive!"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5555)
