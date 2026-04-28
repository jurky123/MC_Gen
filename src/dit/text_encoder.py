import torch
from transformers import CLIPTokenizer, CLIPTextModel

_tokenizer = None
_text_encoder = None


def _load_models():
    """延迟加载，避免 import 时自动下载模型"""
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
    _text_encoder.to(device)
    with torch.no_grad():
        tokens = _tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        output = _text_encoder(**tokens)
    return output.last_hidden_state
