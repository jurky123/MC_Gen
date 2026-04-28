import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import math
"""
DiT (Diffusion Transformer) 模型定义。

整体结构：
  DiT
  ├── patch_embed        — 卷积将图片切为 patch 序列
  ├── pos_embed          — 可学习的位置编码，加在 patch 序列上
  ├── time_embedding     — 时间步嵌入的两层 MLP
  ├── text_proj          — 文本编码维度投影，映射到 embed_dim
  ├── blocks (×N)        — 多个 DiT_block 串联
  ├── final_norm         — 输出前的 LayerNorm
  ├── final_proj         — 将序列投影回像素空间
  └── encode_time_stamp  — 正弦余弦时间步编码

每个 DiT_block 内部：
  norm1 → AdaLN → self_attention  → +残差
  norm2 → AdaLN → cross_attention → +残差
  norm3 → AdaLN → FFN             → +残差

张量格式统一使用 batch_first：序列张量为 [B, L, D]。
"""


class DiT_block(nn.Module):
    """
    DiT 的基本 Transformer 块。
    包含 self-attention、cross-attention、FFN 三个子层，
    每个子层前用 AdaLN 注入时间步信息。
    """
    def __init__(self, config):
        super(DiT_block, self).__init__()
        # 自注意力：patch 序列内部交互
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.embed_dim, num_heads=config.num_heads,
            dropout=config.dropout, batch_first=True
        )
        # 交叉注意力：patch 序列与文本编码交互，K/V 来自 text_encoding
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.embed_dim, num_heads=config.num_heads,
            dropout=config.dropout, batch_first=True
        )
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, config.ffn_dim),
            nn.GELU(),
            nn.Linear(config.ffn_dim, config.embed_dim),
        )
        # 三个 LayerNorm（elementwise_affine=False，因为缩放/偏移由 AdaLN 提供）
        self.norm1 = nn.LayerNorm(config.embed_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(config.embed_dim, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(config.embed_dim, elementwise_affine=False)
        # AdaLN 调制层：从时间步嵌入生成 6×embed_dim 的参数
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.embed_dim, 6 * config.embed_dim)
        )

    def forward(self, x, text_encoding, t_emb):
        """
        Args:
            x:              patch 序列         [B, L, embed_dim]
            text_encoding:  文本编码           [B, T, embed_dim]
            t_emb:          时间步嵌入（顶层已算好）[B, embed_dim]
        Returns:
            x: 经过本块处理后的 patch 序列  [B, L, embed_dim]
        """
        # 从时间步嵌入生成 6 组调制参数
        modulation = self.adaLN_modulation(t_emb)  # [B, 6*embed_dim]
        scale1, shift1, scale2, shift2, scale3, shift3 = modulation.chunk(6, dim=-1)

        # ---- self-attention ----
        normed = self.norm1(x)
        normed = normed * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        attn_out, _ = self.self_attention(normed, normed, normed)
        x = x + attn_out

        # ---- cross-attention ----
        normed = self.norm2(x)
        normed = normed * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        cross_out, _ = self.cross_attention(normed, text_encoding, text_encoding)
        x = x + cross_out

        # ---- FFN ----
        normed = self.norm3(x)
        normed = normed * (1 + scale3.unsqueeze(1)) + shift3.unsqueeze(1)
        x = x + self.ffn(normed)

        return x


class DiT(nn.Module):
    """
    DiT 顶层模型。
    负责 patch 嵌入、位置编码、时间步编码、文本投影、串联 blocks、输出投影。
    """
    def __init__(self, config):
        super(DiT, self).__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.patch_size = config.patch_size
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.out_patch_dim = config.patch_size * config.patch_size * config.in_channels

        # 将图片切分为 patch 序列：[B, C, H, W] → [B, embed_dim, h, w] → [B, num_patches, embed_dim]
        self.patch_embed = nn.Conv2d(
            config.in_channels, config.embed_dim,
            kernel_size=config.patch_size, stride=config.patch_size
        )
        # 可学习的位置编码
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, config.embed_dim) * 0.02
        )
        # 时间步嵌入：两层 MLP
        self.time_embedding = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.SiLU(),
            nn.Linear(config.embed_dim, config.embed_dim),
        )
        # 文本编码投影：将外部编码器输出维度映射到 embed_dim
        self.text_proj = nn.Linear(config.text_dim, config.embed_dim)
        # 堆叠 N 个 DiT_block
        self.blocks = nn.ModuleList([DiT_block(config) for _ in range(config.num_layers)])
        # 输出层
        self.final_norm = nn.LayerNorm(config.embed_dim, elementwise_affine=False)
        self.final_proj = nn.Linear(config.embed_dim, self.out_patch_dim)

    def encode_time_stamp(self, step):
        """
        正弦余弦位置编码风格的时间步编码。
        Args:
            step: 时间步，形状 [B]
        Returns:
            emb: 时间步嵌入 [B, embed_dim]
        """
        half = self.config.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=step.device) / half
        )
        angles = step[:, None].float() * freqs[None, :]
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        emb = self.time_embedding(emb)
        return emb

    def unpatchify(self, x):
        """
        将 patch 序列还原为图片。
        [B, num_patches, out_patch_dim] → [B, C, H, W]
        """
        B, L, _ = x.shape
        h = w = int(L ** 0.5)
        x = x.reshape(B, h, w, self.patch_size, self.patch_size, self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(B, self.in_channels, h * self.patch_size, w * self.patch_size)
        return x

    def forward(self, x, text_encoding, time_step):
        """
        Args:
            x:              图片                [B, C, H, W]
            text_encoding:  文本编码            [B, T, text_dim]
            time_step:      扩散时间步           [B]
        Returns:
            x: 模型预测的噪声 / v-prediction   [B, C, H, W]
        """
        # Patch 嵌入
        x = self.patch_embed(x)                                 # [B, D, h, w]
        x = x.flatten(2).transpose(1, 2)                       # [B, num_patches, D]
        x = x + self.pos_embed                                 # 位置编码只加一次

        t_emb = self.encode_time_stamp(time_step)              # [B, D]
        text_encoding = self.text_proj(text_encoding)          # [B, T, D]

        for block in self.blocks:
            x = block(x, text_encoding, t_emb)

        x = self.final_norm(x)
        x = self.final_proj(x)                                 # [B, num_patches, out_patch_dim]
        x = self.unpatchify(x)                                 # [B, C, H, W]
        return x

    def save(self, path):
        """保存模型参数到指定路径"""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """从指定路径加载模型参数"""
        self.load_state_dict(torch.load(path))
