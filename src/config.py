"""
存放模型参数以及训练超参数。
"""

class ModelConfig:
    # ---- DiT 结构 ----
    embed_dim = 512          # 隐藏层维度
    num_heads = 8            # 注意力头数
    num_layers = 12          # DiT_block 层数
    ffn_dim = 2048           # FFN 中间层维度
    dropout = 0.1            # attention dropout

    # ---- 图像 ----
    image_size = 16          # MC 材质通常 16×16
    patch_size = 1           # 每个 patch 的大小
    in_channels = 4          # RGBA

    # ---- 文本 ----
    text_dim = 512           # CLIP ViT-B/32 输出维度


class TrainingConfig:
    # ---- 扩散 ----
    num_timesteps = 1000     # 扩散总步数
    beta_start = 1e-4        # β 起始值
    beta_end = 0.02          # β 终止值

    # ---- 训练 ----
    batch_size = 32
    learning_rate = 2e-4
    epochs = 2000

    # ---- 路径 ----
    data_dir = "data/"
    checkpoint_dir = "checkpoints/"
    save_every = 10          # 每隔多少 epoch 保存一次
