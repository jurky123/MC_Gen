import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import logging
from transformers import CLIPTokenizer, CLIPTextModel

# 抑制 transformers 模型加载时的 UNEXPECTED keys 警告（CLIPTextModel 只取文本权重，视觉权重自然跳过）
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

_tokenizer = None
_text_encoder = None


def _load_models():
    """延迟加载，首次调用时下载并缓存模型"""
    global _tokenizer, _text_encoder
    if _tokenizer is None:
        _tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        _text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        _text_encoder.eval()
        for p in _text_encoder.parameters():
            p.requires_grad = False


def encode_text(texts, device="cpu"):
    """
    将文本列表编码为 hidden states，用于 cross-attention。

    Args:
        texts: str 或 list[str]
        device: 输出张量所在设备
    Returns:
        last_hidden_state: [B, T, 512]
    """
    _load_models()
    # 仅在设备变化时才移动，避免每 batch 重复搬运
    if str(_text_encoder.device) != str(device):
        _text_encoder.to(device)
    with torch.no_grad():
        tokens = _tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        output = _text_encoder(**tokens)
    return output.last_hidden_state
