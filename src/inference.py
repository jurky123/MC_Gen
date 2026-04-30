"""
负责实际推理
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from torchvision.utils import save_image

from config import ModelConfig, TrainingConfig
from dit.model import DiT
from dit.diffusion import Diffusion
from dit.text_encoder import encode_text


def main():
    parser = argparse.ArgumentParser(description="DiT 材质生成推理")
    parser.add_argument("--prompt", "-p", type=str, default="fire_blade",
                        help="文本描述")
    parser.add_argument("--checkpoint", "-c", type=str, default="checkpoints/dit_epoch14.pth",
                        help="模型检查点路径")
    parser.add_argument("--output", "-o", type=str, default="output.png",
                        help="输出图片路径")
    parser.add_argument("--cfg_scale", "-g", type=float, default=1.5,
                        help="CFG 引导强度")
    parser.add_argument("--steps", "-s", type=int, default=None,
                        help="采样步数（默认等于训练时的 num_timesteps，设小值可用 DDIM 加速）")
    parser.add_argument("--cpu", action="store_true",
                        help="强制使用 CPU 推理")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    # ---- 从检查点恢复 config ----
    ckpt = torch.load(args.checkpoint, map_location=device)

    model_cfg = ModelConfig()
    if "model_config" in ckpt:
        for k, v in ckpt["model_config"].items():
            setattr(model_cfg, k, v)
        print("Loaded model config from checkpoint")

    train_cfg = TrainingConfig()
    if "train_config" in ckpt:
        for k, v in ckpt["train_config"].items():
            setattr(train_cfg, k, v)
        print(f"Loaded train config from checkpoint (num_timesteps={train_cfg.num_timesteps})")

    # ---- 加载模型 ----
    model = DiT(model_cfg).to(device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    # ---- 采样 ----
    with torch.no_grad():
        if args.steps is not None and args.steps != train_cfg.num_timesteps:
            # 使用 DDIM 子序列采样（跳步加速）
            print(f"DDIM sampling: {args.steps} steps (trained on {train_cfg.num_timesteps})")
            diffusion = Diffusion(train_cfg, model)
            img = ddim_sample(
                diffusion, model, model_cfg, args.prompt,
                device, args.steps, args.cfg_scale
            )
        else:
            diffusion = Diffusion(train_cfg, model)
            text_encoding = encode_text([args.prompt], device=device)
            img_shape = (1, model_cfg.in_channels, model_cfg.image_size, model_cfg.image_size)
            print(f"Prompt: {args.prompt}")
            print(f"DDPM sampling {train_cfg.num_timesteps} steps, CFG={args.cfg_scale} ...")
            img = diffusion.sample(text_encoding, img_shape, cfg_scale=args.cfg_scale)

    # ---- 后处理 & 保存 ----
    # 从 [-1, 1] 映射回 [0, 1]（对应训练时的 Normalize(0.5, 0.5)）
    img = img * 0.5 + 0.5
    print(f"Output stats — min: {img.min().item():.4f}, max: {img.max().item():.4f}, mean: {img.mean().item():.4f}")
    img = img.clamp(0, 1)
    save_image(img, args.output)
    print(f"Saved to {args.output}")


def ddim_sample(diffusion, model, model_cfg, prompt, device, steps, cfg_scale):
    """
    DDIM 确定性采样，用更少的步数加速推理。

    从训练时的 T 步中均匀抽取 steps 个子步，
    每步之间直接跳跃，不需要每步都算。
    """
    text_encoding = encode_text([prompt], device=device)

    # 从训练步数中均匀选取子序列
    T = diffusion.num_timesteps
    timesteps = torch.linspace(T - 1, 0, steps, dtype=torch.long, device=device)

    x = torch.randn(1, model_cfg.in_channels, model_cfg.image_size, model_cfg.image_size,
                     device=device)

    for i in range(len(timesteps)):
        t = timesteps[i]
        t_batch = torch.tensor([t], device=device, dtype=torch.long)

        # CFG 噪声预测
        if cfg_scale != 1.0:
            null_encoding = torch.zeros_like(text_encoding)
            noise_cond = model(x, text_encoding, t_batch)
            noise_uncond = model(x, null_encoding, t_batch)
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = model(x, text_encoding, t_batch)

        # 预测 x_0
        alpha_bar_t = diffusion.alphas_cumprod[t]
        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        x0_pred = x0_pred.clamp(-1, 1)  # 干净图片不应超出归一化范围，防止误差放大

        # 下一步的 α̅
        t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(-1, device=device)
        if t_next < 0:
            x = x0_pred  # 最后一步，直接输出 x_0 预测
        else:
            alpha_bar_next = diffusion.alphas_cumprod[t_next]
            # DDIM 确定性更新（σ=0）
            x = torch.sqrt(alpha_bar_next) * x0_pred + \
                torch.sqrt(1 - alpha_bar_next) * noise_pred

        if i % max(1, steps // 5) == 0 or i == len(timesteps) - 1:
            print(f"  step {i+1:3d}/{steps}  t={t.item():4d}  "
                  f"x min={x.min().item():.3f}  max={x.max().item():.3f}  mean={x.mean().item():.3f}")

    return x


if __name__ == "__main__":
    main()
