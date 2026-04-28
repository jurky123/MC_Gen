import torch
import torch.nn as nn


class Diffusion:
    """
    DDPM 扩散/去噪流程。

    前向加噪：x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
    反向去噪：模型预测噪声 ε_θ(x_t, t)，然后逐步去噪
    """
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.num_timesteps = config.num_timesteps

        # ---- 噪声调度：线性 β schedule ----
        # β_t 从 beta_start 线性增长到 beta_end
        betas = torch.linspace(
            config.beta_start, config.beta_end, config.num_timesteps
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # α̅_t

        # 注册为 buffer（不参与梯度，但随模型保存/加载）
        setattr(self, 'betas', betas)
        setattr(self, 'alphas', alphas)
        setattr(self, 'alphas_cumprod', alphas_cumprod)

        # 反向过程会用到的预计算量
        setattr(self, 'sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        setattr(self, 'sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))


    def add_noise(self, x, time_step):
        """
        前向扩散：对干净图片 x_0 加噪，得到 x_t。

        公式：x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε

        Args:
            x:         干净图片 / latent     [B, C, H, W]
            time_step: 时间步索引            [B]，取值 0 ~ num_timesteps-1
        Returns:
            x_t:   加噪后的图片   [B, C, H, W]
            noise: 添加的噪声     [B, C, H, W]（作为训练标签）
        """
        noise = torch.randn_like(x)

        # 取出每个样本对应时间步的系数，reshape 为 [B, 1, 1, 1] 以便广播
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[time_step]          # [B]
        sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1, 1)            # [B, 1, 1, 1]

        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[time_step]
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1, 1)

        # 加权融合：信号衰减 + 噪声注入
        x_t = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
        return x_t, noise

    def step(self, x, time_step, text_encoding, cfg_scale=3.0):
        """
        单步反向去噪（DDPM 采样步）。

        Args:
            x:              当前噪声图片    [B, C, H, W]
            time_step:      当前时间步      [B]
            text_encoding:  文本编码        [B, T, text_dim]
            cfg_scale:      CFG 引导强度
        Returns:
            x_prev: 去噪一步后的图片 [B, C, H, W]
        """
        # CFG：无条件 + 有条件预测，加权融合
        # 无条件：text_encoding 置零
        if cfg_scale != 1.0:
            null_encoding = torch.zeros_like(text_encoding)
            noise_cond = self.model(x, text_encoding, time_step)
            noise_uncond = self.model(x, null_encoding, time_step)
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = self.model(x, text_encoding, time_step)

        # 系数
        alpha = self.alphas[time_step].view(-1, 1, 1, 1)
        alpha_cumprod = self.alphas_cumprod[time_step].view(-1, 1, 1, 1)
        beta = self.betas[time_step].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[time_step].view(-1, 1, 1, 1)

        # 预测原始 x_0
        x0_pred = (x - sqrt_one_minus_alpha_cumprod * noise_pred) / torch.sqrt(alpha_cumprod)

        # DDPM 均值公式
        coeff_x0 = (torch.sqrt(alpha_cumprod / alpha) * beta) / (1 - alpha_cumprod)
        coeff_xt = torch.sqrt(alpha) * (1 - alpha_cumprod / alpha) / (1 - alpha_cumprod)

        mean = coeff_x0 * x0_pred + coeff_xt * x

        # 加方差噪声（最后一步不加）
        noise = torch.randn_like(x)
        sigma = torch.sqrt(beta)
        mask = (time_step > 0).float().view(-1, 1, 1, 1)
        x_prev = mean + mask * sigma * noise

        return x_prev

    def sample(self, text_encoding, image_shape, cfg_scale=3.0):
        """
        完整采样：从纯噪声开始，逐步去噪得到最终图片。

        Args:
            text_encoding:  文本编码            [B, T, text_dim]
            image_shape:    输出图片形状         (B, C, H, W)
            cfg_scale:      CFG 引导强度
        Returns:
            x: 去噪后的最终图片 [B, C, H, W]
        """
        x = torch.randn(image_shape, device=text_encoding.device)

        for t in reversed(range(self.num_timesteps)):
            time_step = torch.full((image_shape[0],), t, device=text_encoding.device, dtype=torch.long)
            x = self.step(x, time_step, text_encoding, cfg_scale)

        return x
