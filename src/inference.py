"""
负责实际推理
"""
from dit.model import DiT
from dit.diffusion import Diffusion
import torch
from config import TrainingConfig, ModelConfig
from dit.text_encoder import encode_text
def main():
    #读取一下参数，包括图片尺寸，prompt，输出位置，引导系数，时间步数等，从命令行读取
    text = "diamond sword"#先用一个默认的放在这
    img_shape = (1, 4, 16, 16)
    config = ModelConfig()
    model = DiT(config)

    model.load_state_dict(torch.load("model.pth"))
    diffusion = Diffusion(config, model)
    text_encoding = encode_text(text)
    img = diffusion.sample(text_encoding,img_shape, cfg_scale=3.0, steps=50)
    #保存图片，到输出位置

    