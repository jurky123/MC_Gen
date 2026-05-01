"""
负责训练模型

单卡:  python train.py
多卡:  torchrun --nproc_per_node=N train.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
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
            transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transform(self.images[idx].convert("RGBA"))
        label = self.labels[idx]
        return image, label


def setup_dist():
    """初始化分布式环境，单卡时返回 rank=0, world_size=1"""
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        return rank, world_size
    return 0, 1


def cleanup_dist():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


# ──────────────────── 主函数 ────────────────────

def main():
    rank, world_size = setup_dist()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if is_main(rank):
        print(f"Using {world_size} GPU(s)")

    import argparse
    parser = argparse.ArgumentParser(description="DiT 训练")
    parser.add_argument("--resume", "-r", type=str, default=None,
                        help="从指定 checkpoint 恢复训练（例如 checkpoints/dit_epoch100.pth）")
    args = parser.parse_args()

    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()

    # ---- 模型（先建裸模型，resume 加载权重后再包 DDP） ----
    model = DiT(model_cfg).to(device)
    start_epoch = 1

    if args.resume is not None:
        if is_main(rank):
            print(f"Resuming from {args.resume} ...")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        if is_main(rank):
            print(f"  restored epoch={ckpt['epoch']}, loss={ckpt.get('loss', 'N/A')}, "
                  f"resuming from epoch {start_epoch}")

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    diffusion = Diffusion(train_cfg, model)

    # ---- 优化器 ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)
    if args.resume is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # ---- DataLoader ----
    images, labels = load_data()
    dataset = LoadedDataset(images, labels, image_size=model_cfg.image_size)
    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )

    if is_main(rank):
        os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    encode_text(["warmup"], device=device)

    # ============================================================
    # 训练循环
    # ============================================================
    for epoch in range(start_epoch, train_cfg.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)  # 确保每 epoch 的数据 shuffle 不同

        total_loss = 0.0

        for images, texts in dataloader:
            images = images.to(device)
            text_encoding = encode_text(list(texts), device=device)

            time_step = torch.randint(
                0, train_cfg.num_timesteps, (images.shape[0],),
                device=device
            )

            x_t, noise = diffusion.add_noise(images, time_step)
            noise_pred = model(x_t, text_encoding, time_step)
            loss = nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 跨卡汇总 loss
        if world_size > 1:
            loss_tensor = torch.tensor(total_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            total_loss = loss_tensor.item()

        avg_loss = total_loss / (len(dataloader) * world_size)

        if is_main(rank):
            print(f"Epoch {epoch:3d}/{train_cfg.epochs}  |  loss = {avg_loss:.6f}")

        # ---- 保存检查点（仅 rank 0） ----
        if epoch % train_cfg.save_every == 0 and is_main(rank):
            # DDP 模型需取 .module 获取原始 state_dict
            state_dict = model.module.state_dict() if world_size > 1 else model.state_dict()
            ckpt_path = os.path.join(train_cfg.checkpoint_dir, f"dit_epoch{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": state_dict,
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

    cleanup_dist()
    if is_main(rank):
        print("Training done.")


if __name__ == "__main__":
    main()
