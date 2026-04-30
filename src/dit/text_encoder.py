import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import logging
from transformers import CLIPTokenizer, CLIPTextModel

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

_tokenizer = None
_text_encoder = None

# 优先使用本地离线模型（checkpoints/models--openai--clip-vit-base-patch32/），
# 不存在时自动从镜像站下载。
# huggingface hub 下载的目录结构为 snapshots/<hash>/，模型文件在 snapshot 里，
# 需往里找一层；如果是直接放模型文件的目录则直接用。
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_LOCAL_DIR = os.path.join(_project_root, "checkpoints", "models--openai--clip-vit-base-patch32")


def _resolve_model_path(base_dir):
    """解析本地模型目录：如果是 hub snapshot 结构则深入一层，否则直接用"""
    if not os.path.isdir(base_dir):
        return None
    if os.path.isfile(os.path.join(base_dir, "config.json")):
        return base_dir
    snapshots = os.path.join(base_dir, "snapshots")
    if os.path.isdir(snapshots):
        for name in os.listdir(snapshots):
            sub = os.path.join(snapshots, name)
            if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, "config.json")):
                return sub
    return None


MODEL_NAME = _resolve_model_path(_LOCAL_DIR) or "openai/clip-vit-base-patch32"


def _load_models():
    """延迟加载，首次调用时下载并缓存模型"""
    global _tokenizer, _text_encoder
    if _tokenizer is None:
        print(f"Loading CLIP from: {MODEL_NAME}")
        _tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)
        _text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME)
        _text_encoder.eval()
        for p in _text_encoder.parameters():
            p.requires_grad = False


def encode_text(texts, device="cpu"):
    """
    将文本列表编码为 hidden states，用于 cross-attention。

    下划线会被替换为空格 —— CLIP 的 BPE tokenizer 将 _ 视为独立 token，
    而空格才是 CLIP 训练数据中的正常词分隔符。"iron_bow" 替换为 "iron bow" 后
    分词质量和 embedding 语义更准确。

    Args:
        texts: str 或 list[str]
        device: 输出张量所在设备
    Returns:
        last_hidden_state: [B, T, 512]
    """
    _load_models()
    if isinstance(texts, str):
        texts = texts.replace("_", " ")
    else:
        texts = [t.replace("_", " ") for t in texts]
    if str(_text_encoder.device) != str(device):
        _text_encoder.to(device)
    with torch.no_grad():
        tokens = _tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        output = _text_encoder(**tokens)
    return output.last_hidden_state
