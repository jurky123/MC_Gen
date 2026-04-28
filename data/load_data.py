"""导入读取图片，读取json相关的库"""
import json
import os
from PIL import Image

IMG_DIR = os.path.join("data", "dataset", "images")
LABEL_FILE = os.path.join("data", "dataset", "labels.jsonl")


def load_data(batch_size=None):
    # 读取dataset文件夹下的图片文件
    image_files = sorted(
        f for f in os.listdir(IMG_DIR) if f.endswith(".png")
    )

    # 读取dataset文件夹下的labels.jsonl文件，提取文件名→标签和类别映射
    with open(LABEL_FILE, "r", encoding="utf-8") as f:
        records = [
            json.loads(line)
            for line in f if line.strip()
        ]
        label_map = {r["filename"]: r["label"] for r in records}
        category_map = {r["filename"]: r["category"] for r in records}

    # 过滤：只保留 category 为 "item" 的图片
    image_files = [fn for fn in image_files if category_map.get(fn) == "item"]
    # 截取指定数量
    image_files = image_files[:batch_size]

    # 构建图片
    images = [Image.open(os.path.join(IMG_DIR, fn)) for fn in image_files]

    # 构建标签列表
    labels = [label_map[fn] for fn in image_files]

    return images, labels
