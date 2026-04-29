"""从 parquet 数据集加载图片和标签"""
import os
from io import BytesIO
from pathlib import Path
from PIL import Image
import pandas as pd

DATA_DIR = os.path.join("data", "dataset", "data_large")


def load_data(batch_size=None):
    # 读取所有 parquet 文件
    dfs = []
    for fn in sorted(Path(DATA_DIR).glob("*.parquet")):
        dfs.append(pd.read_parquet(fn))
    df = pd.concat(dfs, ignore_index=True)

    # 只保留 type 为 "item" 的数据
    df = df[df["type"] == "item"]
    df = df.reset_index(drop=True)

    # 截取指定数量
    if batch_size is not None:
        df = df.head(batch_size)

    # 构建图片列表：从 bytes 解码为 PIL Image
    images = []
    for img_data in df["image"]:
        img = Image.open(BytesIO(img_data["bytes"]))
        images.append(img)

    # 标签列表：文件名去掉扩展名
    labels = [Path(fn).stem for fn in df["file_name"]]

    return images, labels
