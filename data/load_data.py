"""导入读取图片，读取json相关的库"""
import json
import os
from PIL import Image


def load_data(batch_size=None):
    # 读取dataset文件夹下的图片文件，batch_size是最大读取数量
    image_files = sorted(
        f for f in os.listdir("dataset/images") if f.endswith(".png")
    )[:batch_size]

    # 读取dataset文件夹下的labels.jsonl文件
    with open("dataset/labels.jsonl", "r", encoding="utf-8") as f:
        label_map = {
            item["filename"]: item["label"]
            for line in f if line.strip()
            for item in [json.loads(line)]
        }

    # 构建x：图片文件的内容
    images = [Image.open(os.path.join("dataset/images", fn)) for fn in image_files]

    # 构建y：将图片文件名在labels.jsonl中查找对应的标签
    labels = [label_map[fn] for fn in image_files]

    return images, labels
