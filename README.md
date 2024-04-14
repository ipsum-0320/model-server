# 安装依赖

首先最好创建一个虚拟环境:

```python
python3 -m venv .venv
```
然后激活虚拟环境:

```python
. .venv/bin/activate
```

退出虚拟环境使用:

```python
deactivate
```

之后安装依赖即可。
```python
pip install -r requirements.txt
```

> 导出依赖使用 pip freeze > requirements.txt

# 使用

如下命令可以启动 server。

```python
python server.py
```

如果想要切换模型，可以修改 checkpoints 中的 *.pth 文件。