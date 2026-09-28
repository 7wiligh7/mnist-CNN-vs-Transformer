# 手写数字识别：CNN 与 Vision Transformer 两种方案对比

同一个任务（MNIST 手写数字 10 分类），用两种截然不同的深度学习范式各实现一遍，并对比它们的表现。

| 方案 A：卷积神经网络（CNN） | 方案 B：Vision Transformer（ViT） |
| :---: | :---: |
| [`cnn/`](cnn/) | [`transformer/`](transformer/) |

**这个仓库想回答一个问题**：在 28×28 的小尺寸灰度图上，2020 年后才火起来的 Transformer 架构，真的能取代统治图像领域十余年的 CNN 吗？

答案是：**在小数据集上不能**。下面用实测数据说明原因。

---

## 实测结果对比

以下数据均为本机在 Intel CPU、无 GPU 环境下实际运行所得。

| 指标 | CNN 方案 | ViT 方案 | 对比 |
| :--- | ---: | ---: | :--- |
| **测试集准确率** | **98.15%** | **97.96%** | CNN 高 0.19% |
| 模型参数量 | 63,626 | 205,066 | ViT 是 CNN 的 **3.2 倍** |
| 网络深度 | 1 个卷积层 | 4 个 Transformer 块 | — |
| 训练轮数 | 10 epochs | 10 epochs | 相同 |
| 优化器 | Adam（lr = 0.01） | Adam（lr = 0.001） | — |
| 最后一轮训练损失 | — | 0.0694 | — |
| **CPU 训练总耗时** | **140 秒** | **472 秒** | ViT 慢 **3.4 倍** |
| 权重文件体积 | 253 KB | 820 KB | — |

> **关于数据说明**
> - 训练耗时为两个方案在同一会话中连续测得（数据均已缓存），计时区间：CNN 140~240 秒，ViT 472~782 秒。不同机器负载下波动较大，倍率为多次实测的稳定区间（3.4~3.5 倍）。
> - 准确率为多次运行结果，重复运行可复现（PyTorch 默认随机种子固定，训练过程确定）。
> - 全程 CPU 训练，未使用 GPU。ViT 在 CPU 上训练需 8~13 分钟，CNN 约 2~4 分钟。

**一句话结论**：ViT 用了 3.2 倍的参数、3.4 倍的算力，准确率反而略低于一个只有一层的轻量 CNN。

---

## 为什么 Transformer 在这里输了

这不是实现问题，而是**架构与数据规模不匹配**。

### 核心差异：归纳偏置（Inductive Bias）

> **归纳偏置** = 模型对「什么样的输入应该产生什么样的输出」的先验假设。假设越强，模型需要从数据中学习的东西越少。

|  | CNN | ViT |
| :--- | :--- | :--- |
| 归纳偏置 | **强** | **弱** |
| 内置假设 | 局部性（相邻像素相关）<br>平移不变性（平移不影响类别）<br>层级抽象（边缘 → 纹理 → 部件） | 几乎没有，只知道「序列中各元素存在关联」 |
| 位置信息 | 天然编码在卷积核的局部感受野中 | 必须**显式注入**位置编码 |
| 数据需求 | 少样本也能学得不错 | 需要海量数据才能「学出」位置与层级关系 |

关键在于：

- **CNN 开箱即用地内置了「图像是空间局部结构」这一先验**。卷积核只在 5×5 的小窗口内滑动，等于免费获得了「相邻像素相关」和「平移不变」两个强假设。卷积核权重共享，则天然保证了平移不变性。
- **ViT 把图像切成 16 个 patch 扔进自注意力后，模型对「这是一个图像」几乎一无所知**。它必须额外学习位置编码（所以有 `pos_embed`），还必须从数据中自己发现局部性与层级结构。

ViT 的自注意力擅长捕捉**全局依赖**——这在 ImageNet 级别的自然图像上价值巨大。但 MNIST 是 28×28 的单通道二值图，笔画简单、类内差异小、全局依赖弱，CNN 的先验恰好完全对口，于是 CNN 用 1/3 的参数和 1/4 的时间取得了更好的结果。

> ViT 的真正优势要在**大数据 + 大算力**下才显现：论文中 ViT-Large 在 ImageNet-21k 上预训练后达到 88.55% 的 ImageNet 准确率，远超同规模的 CNN。**本仓库的规模远未触及那个量级。**

### 训练效率的另一层原因

- CNN 只有一个 `5×5` 卷积层，参数量少，梯度路径短，CPU 上跑 10 轮约 2~4 分钟。
- ViT 有 4 层 Transformer，每层包含多头自注意力（计算量随序列长度平方增长）和两层 MLP。注意力矩阵的计算无法像卷积那样充分复用硬件，导致在 CPU 上训练成本高得多。

---

## 架构对比

### CNN 方案（`cnn/`）

```
输入 (1, 28, 28)
   │
   ├─ Conv2d(1 → 32, kernel_size=5, padding=2)   32 个 5×5 卷积核
   ├─ BatchNorm2d(32)                             批归一化，稳定梯度
   ├─ ReLU()                                      激活函数
   └─ MaxPool2d(2)                                最大池化，28×28 → 14×14
   │
Flatten (32 × 14 × 14 = 6,272)
   │
   └─ Linear(6,272 → 10)                         全连接分类
   │
输出 (10) —— logits，未经 softmax
```

一个卷积层 + 一个全连接层，共 63,626 个参数。结构极简，聚焦于理解卷积、批归一化、池化三个核心组件的作用。

### ViT 方案（`transformer/`）

```
输入 (B, 1, 28, 28)
   │
   ├─ PatchEmbedding
   │    └─ Conv2d(1 → 64, k=7, stride=7)          切分为 4×4=16 个 patch 并投影
   │    └─ flatten(2).transpose(1, 2)              (B, 64, 4, 4) → (B, 16, 64)
   │
   ├─ 拼接可学习 CLS token                        (B, 16, 64) → (B, 17, 64)
   ├─ 加上可学习位置编码 pos_embed                 注入位置信息
   ├─ Dropout
   │
   ├─ TransformerBlock × 4
   │    ├─ norm1 → MultiheadAttention(4 heads) → 残差相加
   │    └─ norm2 → MLP(64 → 256 → 64)            GELU 激活
   │         → 残差相加
   │
   ├─ LayerNorm                                   最终归一化
   │
   └─ Head: Linear(64 → 10)                       取 CLS token 输出做分类
   │
输出 (B, 10) —— logits，未经 softmax
```

超参数：patch 7×7、嵌入维度 64、编码器层数 4、注意力头数 4、MLP 扩张倍数 4.0、dropout 0.1。共 205,066 个参数。

采用 **Pre-Norm** 结构（先归一化再进子层），比原论文的 Post-Norm 训练更稳定。

### 逐组件对照

| 组件 | CNN 方案 | ViT 方案 |
| :--- | :--- | :--- |
| 特征提取 | 3 层卷积 + BN + ReLU | Patch Embedding（卷积仅用于切分图像） |
| 降采样 | MaxPool 降 2× | 无，patch 切分本身即为 4× 降采样 |
| 位置信息 | 隐含于卷积局部感受野 | 显式可学习 `pos_embed` |
| 全局聚合 | 展平后全连接 | CLS token + 自注意力 |
| 归一化 | BatchNorm | LayerNorm（Pre-Norm） |
| 激活函数 | ReLU | GELU |
| 残差连接 | 无 | 8 处（4 块 × 2 处） |
| 正则化 | 无显式 dropout | Dropout 0.1 |
| 深度调节手段 | 增加卷积层 | `depth` / `num_heads` / `embed_dim` |

---

## 项目结构

```
mnist-CNN-vs-Transformer/
├── README.md               # 本文件：两种方案的对比与总览
├── requirements.txt        # 依赖声明（两方案共用）
├── .gitignore              # 忽略数据集、模型权重、缓存等产物
│
├── cnn/                    # ── 方案 A：卷积神经网络 ──
│   ├── model.py            #   模型定义
│   ├── train_cnn.py        #   训练 + 评估 + 保存权重
│   ├── visualize.py        #   可视化测试（OpenCV 逐张弹窗）
│   ├── predict.py          #   识别自定义图片
│   ├── my_digit.png        #   样例测试图片
│   └── README.md           #   CNN 方案详解
│
└── transformer/            # ── 方案 B：Vision Transformer ──
    ├── model.py            #   模型定义：PatchEmbedding / TransformerBlock / ViT
    ├── train.py            #   训练
    ├── evaluate.py         #   测试集评估
    ├── data_loader.py      #   数据加载与预处理
    └── README.md           #   ViT 方案详解
```

> 每个子目录的 README 有该方案的完整说明：逐层参数解释、关键设计理由、运行方式、常见问题。

---

## 快速开始

### 1. 环境准备

```bash
# 建议使用 Python 3.9+
pip install -r requirements.txt
```

> `torch` / `torchvision` 请按自己的硬件（CPU / CUDA）选择对应版本，详见 [PyTorch 官方安装指引](https://pytorch.org/get-started/locally/)。
> `opencv-python` 仅 `cnn/visualize.py` 的弹窗可视化需要，不需要可视化的可忽略。

### 2. 运行 CNN 方案

```bash
cd cnn
python train_cnn.py     # 训练（CPU 约 2~4 分钟）
python visualize.py     # 可视化测试（需 GUI 环境）
python predict.py       # 识别自定义图片
```

### 3. 运行 ViT 方案

```bash
cd transformer
python train.py         # 训练（CPU 约 8~13 分钟）
python evaluate.py      # 测试集评估
```

### 4. 数据集

两方案均使用 torchvision 自动下载的 MNIST（60,000 训练 / 10,000 测试，约 76 MB），首次运行自动完成，无需手动处理。

预处理统一为 `ToTensor()` + `Normalize((0.1307,), (0.3081,))`。

---

## 参考资料

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale（ViT 原论文）](https://arxiv.org/abs/2010.11929)
- [Deep Residual Learning for Image Recognition（ResNet）](https://arxiv.org/abs/1512.03385)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [torchvision MNIST 数据集说明](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)
- [PyTorch Pro 诀窍：权重保存与加载](https://pytorch.org/docs/stable/notes/serialization.html)
