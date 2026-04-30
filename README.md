# MC-Gen

基于 DiT (Diffusion Transformer) 生成 Minecraft 风格材质的项目。数据集从公开发布的 mod 中解包。

## 结构

```
src/
├── config.py              # 模型超参与训练参数
├── train.py               # 训练入口
├── inference.py           # 推理入口
└── dit/
    ├── model.py           # DiT 模型定义（AdaLN + self/cross-attn + FFN）
    ├── diffusion.py       # DDPM 扩散/采样流程
    └── text_encoder.py    # 文本编码器
```

## 原理

- **DiT**：用 AdaLN 将时间步信息注入每层 Transformer，采用DiT架构预测噪声
- **Cross-Attention**：将文本描述作为 K/V，指导材质生成方向
- **DDPM**：前向加权加噪 → 反向逐步去噪，CFG 增强文本引导

## 使用
1.构建数据集，可使用huggingface上的数据集，项目提供了一个自己构建的方案（运行modrinth_downloader.py下载jar文件，再运行extract_textures解包出图片，默认只提取item和block）
2.训练python src/train.py训练时默认只使用item进行训练，参数在config.py的TrainConfig中调整。
3.推理python src/inference.py推理参数通过命令行传入。


## 依赖
```
- Python 3.10+
- PyTorch 2.x


```
## 效果

采用单张40GB显存L40训练约12小时，14个epoch后虽然loss还没收敛，但能生成出质量比较好并且符合prompt的图片。
<img src="output.png" width="512" alt="生成效果">

