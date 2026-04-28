import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import math
"""
DiT (Diffusion Transformer) 模型定义。

整体结构：
  DiT
  ├── pos_embed          — 可学习的位置编码，加在 patch 序列上
  ├── time_embedding     — 时间步嵌入的两层 MLP
  ├── text_proj          — 文本编码维度投影，映射到 embed_dim
  ├── blocks (×N)        — 多个 DiT_block 串联
  └── encode_time_stamp  — 正弦余弦时间步编码

每个 DiT_block 内部：
  norm1 → AdaLN → self_attention  → +残差
  norm2 → AdaLN → cross_attention → +残差
  norm3 → AdaLN → FFN             → +残差

AdaLN (Adaptive Layer Normalization)：
  时间步向量 t_emb 经过 adaLN_modulation 生成 6 组 (scale, shift)，
  分别用于三个子层，让时间步信息注入到每一层的归一化中。

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
        # 6 = 3组 (scale, shift)，分别对应 self-attn、cross-attn、FFN
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
        attn_out, _ = self.self_attention(normed, normed, normed)  # Q=K=V=normed
        x = x + attn_out  # 残差连接，加在原始 x 上

        # ---- cross-attention ----
        normed = self.norm2(x)
        normed = normed * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        cross_out, _ = self.cross_attention(normed, text_encoding, text_encoding)  # K/V 来自文本
        x = x + cross_out

        # ---- FFN ----
        normed = self.norm3(x)
        normed = normed * (1 + scale3.unsqueeze(1)) + shift3.unsqueeze(1)
        x = x + self.ffn(normed)

        return x


class DiT(nn.Module):
    """
    DiT 顶层模型。
    负责位置编码、时间步编码、文本维度投影，并串联多个 DiT_block。
    """
    def __init__(self, config):
        super(DiT, self).__init__()
        self.config = config
        # 可学习的位置编码，形状 [1, num_patches, embed_dim]，广播到 batch
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.num_patches, config.embed_dim) * 0.02
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

    def encode_time_stamp(self, step):
        """
        正弦余弦位置编码风格的时间步编码。

        对时间步 t 生成一个 embed_dim 维的向量：
        - 前半维度用 sin，后半维度用 cos
        - 频率按对数尺度从 1 到 1/10000 递减

        Args:
            step: 时间步，形状 [B] 或 [B, 1]
        Returns:
            emb: 时间步嵌入 [B, embed_dim]
        """
        half = self.config.embed_dim // 2
        # 频率：从 1 到 1/10000，在对数空间均匀分布
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=step.device) / half
        )  # [half]
        # 角度 = t * freq，外积得到 [B, half]
        angles = step[:, None].float() * freqs[None, :]  # [B, half]
        # 前半 sin，后半 cos，拼接后经过两层 MLP
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)  # [B, embed_dim]
        emb = self.time_embedding(emb)
        return emb

    def forward(self, x, text_encoding, time_step):
        """
        Args:
            x:              patch 序列          [B, L, embed_dim]
            text_encoding:  文本编码            [B, T, text_dim]
            time_step:      扩散时间步           [B]
        Returns:
            x: 经过所有 DiT_block 处理后的序列 [B, L, embed_dim]
        """
        t_emb = self.encode_time_stamp(time_step)           # [B, embed_dim]
        text_encoding = self.text_proj(text_encoding)       # [B, T, embed_dim]
        x = x + self.pos_embed                              # 位置编码只加一次
        for block in self.blocks:
            x = block(x, text_encoding, t_emb)              # t_emb 传入每个块供 AdaLN 使用
        return x

    def save(self, path):
        """保存模型参数到指定路径"""
        pass

    def load(self, path):
        """从指定路径加载模型参数"""
        pass
