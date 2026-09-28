# 方案 B：手写数字识别（Vision Transformer）

> 本目录是 [mnist-CNN-vs-Transformer](../) 仓库的 **Vision Transformer 方案**。另一方案为 [`cnn/`](../cnn/)（卷积神经网络）。
> **两者的实测对比与差异分析见[主 README](../README.md)。**

从零实现的 Vision Transformer（ViT）图像分类模型，在 MNIST 手写数字数据集上完成 10 分类任务。与常见的 CNN 方案不同，本项目采用**纯 Transformer** 结构——不使用任何卷积层做特征提取（Patch Embedding 层的卷积仅用于切分图像），探索自注意力机制在轻量图像任务上的表现。

## 项目简介

ViT 的核心思想是：把图像切成若干小块（patch），将每个 patch 展平为向量并作为序列元素输入 Transformer，从而让模型通过自注意力机制学习图像各区域之间的全局关联。

本项目完整实现了 ViT 的各个组件，包括 Patch Embedding、可学习位置编码、CLS token、多头自注意力、前馈网络与残差连接，并遵循 **Pre-Norm** 结构以保证训练稳定性。

| 指标 | 数值 |
| :--- | :--- |
| 数据集 | MNIST（60,000 训练 / 10,000 测试） |
| 模型参数量 | 205,066 |
| 训练轮数 | 10 epochs |
| 优化器 | Adam（lr = 0.001） |
| 最后一轮训练损失 | 0.0694 |
| **测试集准确率** | **97.96%** |
| CPU 训练耗时 | 约 472 ~ 782 秒 |

> 上述结果为本机在 Intel CPU、无 GPU 环境下实测所得，重复运行可复现。训练耗时受机器负载影响，波动较大。
>
> 同任务的 CNN 方案（[`cnn/`](../cnn/)）实测为 98.15% 准确率、140~240 秒训练耗时、63,626 参数 —— **本方案参数量是 CNN 的 3.2 倍，训练耗时约 3.4 倍，准确率反而略低**。原因分析见[主 README 的「为什么 Transformer 在这里输了」](../README.md#为什么-transformer-在这里输了)。

## 模型结构

模型超参数：图像 28×28、patch 7×7（切分为 4×4 = 16 个 patch）、嵌入维度 64、编码器层数 4、注意力头数 4、MLP 扩张倍数 4.0、dropout 0.1。

```
输入 (B, 1, 28, 28)
   │
   ├─ PatchEmbedding                              图像分块 + 线性投影
   │    └─ Conv2d(1 → 64, k=7, stride=7)          kernel = stride，实现非重叠切分
   │    └─ flatten(2).transpose(1, 2)              (B, 64, 4, 4) → (B, 16, 64)
   │
   ├─ 拼接 CLS token                               (B, 16, 64) → (B, 17, 64)
   ├─ 加上可学习位置编码 pos_embed                 为序列注入位置信息
   ├─ Dropout
   │
   ├─ TransformerBlock × 4                        残差 + 层归一化
   │    ├─ norm1 → MultiheadAttention(4 heads) → 残差相加
   │    └─ norm2 → MLP(64 → 256 → 64)            GELU 激活
   │         → 残差相加
   │
   ├─ LayerNorm                                   最终层归一化
   │
   └─ Head: Linear(64 → 10)                       取 CLS token 的输出做分类
   │
输出 (B, 10)  —— logits，未经 softmax
```

### 关键设计说明

- **Patch Embedding 用卷积实现**：`kernel_size = stride = patch_size`，等价于一次无重叠的图像切分，同时完成线性投影，因此并不算是「卷积特征提取」，这与 CNN 的本质区别在于后续的建模方式。
- **CLS token**：一个可学习的向量，序列长度为 16 + 1 = 17。其最终输出作为整张图像的聚合表示送入分类头。
- **位置编码**：ViT 中注意力机制本身不感知位置，因此必须显式注入位置信息。这里使用**可学习**（而非正弦固定）位置编码。
- **Pre-Norm 结构**：先归一化再进入子层（`x = x + Attn(norm1(x))`），相比原始论文的 Post-Norm 训练更稳定，不易陷入退化。
- **无残差缩放**：模型规模较小，未引入残差分支上的 LayerScale，结构更简洁。
- **GELU 激活**：前馈网络中使用 GELU 而非 ReLU，曲线更平滑，符合 ViT 原始设计。
- **输出为 logits**：损失函数使用 `CrossEntropyLoss`（内部已含 log_softmax），模型输出层**不接** `Softmax`。
- **权重以 `state_dict` 保存**：只存权重字典而非整个模型对象，便于跨版本迁移、切换设备，并规避 pickle 反序列化执行代码的安全风险。

## 文件结构

```
mnist-CNN-vs-Transformer/
├── README.md           # 仓库主 README：两方案对比与差异分析
├── requirements.txt    # 依赖声明（两方案共用，位于仓库根目录）
├── .gitignore          # 忽略规则（位于仓库根目录）
│
└── transformer/        # ← 本目录
    ├── model.py            # 模型定义：PatchEmbedding / TransformerBlock / VisionTransformer
    ├── train.py            # 训练脚本：训练 10 轮并保存权重
    ├── evaluate.py         # 评估脚本：加载权重，在测试集上计算准确率
    ├── data_loader.py      # 数据加载：数据集下载、归一化预处理、DataLoader 构建
    └── README.md           # 本文件：ViT 方案详解
```

> 以下命令均需在本目录（`transformer/`）下执行。

运行后会自动生成以下路径（均已被 `.gitignore` 排除）：

| 路径 | 说明 |
| :--- | :--- |
| `mnist/` | 自动下载的 MNIST 原始数据，约 76 MB |
| `vit_transformer.pth` | 训练输出的模型权重 `state_dict`，约 820 KB |
| `__pycache__/` | Python 字节码缓存 |

## 快速开始

### 1. 环境准备

```bash
# 建议使用 Python 3.9+
# requirements.txt 位于仓库根目录（两方案共用）
pip install -r ../requirements.txt
```

> 首次 clone 后建议先进入仓库根目录执行 `pip install -r requirements.txt`，再进入本目录运行脚本。
> `torch` 与 `torchvision` 需按自己的硬件（CPU / CUDA）选择对应版本，详见 [PyTorch 官方安装指引](https://pytorch.org/get-started/locally/)。

### 2. 训练模型

```bash
python train.py
```

首次运行会自动下载 MNIST 数据集。每个 epoch 结束后打印平均损失，训练完成后权重保存至 `vit_transformer.pth`，并提示运行 `evaluate.py` 查看准确率。

> 训练轮数可通过 `train.py` 顶部的 `EPOCHS` 常量调整。

### 3. 评估模型

```bash
python evaluate.py
```

加载训练好的权重并在测试集上输出分类准确率。

## 训练过程

在 CPU 环境下的训练损失变化（10 epochs，重复运行可复现）：

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Loss | 0.6130 | 0.2151 | 0.1558 | 0.1299 | 0.1110 | 0.1011 | 0.0903 | 0.0836 | 0.0779 | 0.0694 |

损失下降曲线平滑，说明 Pre-Norm 结构 + Adam 优化器的组合工作良好，无明显震荡或发散。

## 可改进的方向

- **增加验证集与早停机制**：当前训练过程未在每轮结束后评估验证集表现，无法判断模型是否已收敛或过拟合。
- **数据增强**：加入随机旋转、平移等变换，提升泛化能力。
- **更强配置**：增大 `embed_dim`、加深 `depth`、减小 `patch_size`，观察准确率与计算量的权衡。
- **学习率调度**：引入 CosineAnnealingLR 等调度策略进一步收敛。
- **Attention 可视化**：借助权重热力图观察模型关注的图像区域，理解自注意力机制学到了什么。
- **混合精度与批量归一化**：当前 CPU 训练较慢，可考虑 GPU + AMP 加速，或将 LayerNorm 替换为 BatchNorm（Transformer's BN 变体）。

## 参考资料

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale（ViT 原论文）](https://arxiv.org/abs/2010.11929)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [torch.nn.MultiheadAttention 文档](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- [PyTorch Pro 诀窍：权重保存与加载](https://pytorch.org/docs/stable/notes/serialization.html)
