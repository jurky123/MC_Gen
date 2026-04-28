"""
负责训练模型
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from data.load_data import load_data
from config import ModelConfig, TrainingConfig
from dit.model import DiT
from dit.diffusion import Diffusion
from dit.text_encoder import encode_text


class LoadedDataset(Dataset):
    """对 load_data 返回的 (images, labels) 做轻量封装，供 DataLoader 批处理。"""
    def __init__(self, images, labels, image_size):
        self.images = images
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),  # [0,1] → [-1,1]
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transform(self.images[idx].convert("RGBA"))
        label = self.labels[idx]
        return image, label


# ──────────────────── 主函数 ────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()

    # ============================================================
    # 1. 初始化部分（只执行一次）
    # ============================================================

    # ---- 模型 ----
    model = DiT(model_cfg).to(device)
    diffusion = Diffusion(train_cfg, model)  # buffer 自动跟随模型 device

    # ---- 优化器 ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)

    # ---- DataLoader ----
    images, labels = load_data(train_cfg.batch_size)
    dataset = LoadedDataset(images, labels, image_size=model_cfg.image_size)
    dataloader = DataLoader(
        dataset, batch_size=train_cfg.batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )

    # ---- 路径 ----
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    # ---- 预热 text_encoder（首次调用下载并移到 device，后续不再动） ----
    encode_text(["warmup"], device=device)

    # ============================================================
    # 2. 训练循环（epoch × batch）
    # ============================================================
    for epoch in range(1, train_cfg.epochs + 1):
        total_loss = 0.0

        for images, texts in dataloader:
            images = images.to(device)

            # 编码文本
            text_encoding = encode_text(list(texts), device=device)

            # 采样时间步（每个样本随机一个 t）
            time_step = torch.randint(
                0, train_cfg.num_timesteps, (images.shape[0],),
                device=device
            )

            # 加噪
            x_t, noise = diffusion.add_noise(images, time_step)

            # DiT 前向：预测噪声
            noise_pred = model(x_t, text_encoding, time_step)

            # Loss
            loss = nn.functional.mse_loss(noise_pred, noise)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch:3d}/{train_cfg.epochs}  |  loss = {avg_loss:.6f}")

        # ============================================================
        # 3. 保存检查点
        # ============================================================
        if epoch % train_cfg.save_every == 0:
            ckpt_path = os.path.join(train_cfg.checkpoint_dir, f"dit_epoch{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "model_config": {
                    "embed_dim": model_cfg.embed_dim,
                    "num_heads": model_cfg.num_heads,
                    "num_layers": model_cfg.num_layers,
                    "ffn_dim": model_cfg.ffn_dim,
                    "dropout": model_cfg.dropout,
                    "image_size": model_cfg.image_size,
                    "patch_size": model_cfg.patch_size,
                    "in_channels": model_cfg.in_channels,
                    "text_dim": model_cfg.text_dim,
                },
                "train_config": {
                    "num_timesteps": train_cfg.num_timesteps,
                    "beta_start": train_cfg.beta_start,
                    "beta_end": train_cfg.beta_end,
                },
            }, ckpt_path)
            print(f"  -> checkpoint saved: {ckpt_path}")

    print("Training done.")


if __name__ == "__main__":
    main()
