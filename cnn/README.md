# 方案 A：手写数字识别（CNN）

> 本目录是 [mnist-CNN-vs-Transformer](../) 仓库的 **CNN 方案**。另一方案为 [`transformer/`](../transformer/)（Vision Transformer）。
> **两者的实测对比与差异分析见[主 README](../README.md)。**

基于 PyTorch 实现的 MNIST 手写数字识别项目，使用一个轻量级卷积神经网络完成 28×28 灰度手写数字的 10 分类任务。

## 项目简介

CNN 是卷积神经网络在图像分类任务上的经典实现。本项目从零搭建了一个轻量网络结构，覆盖「数据加载 → 模型定义 → 训练 → 测试」完整流程，用于理解卷积层、批归一化、池化与全连接层在图像分类中的作用。

| 指标 | 数值 |
| :--- | :--- |
| 数据集 | MNIST（60,000 训练 / 10,000 测试） |
| 模型参数量 | 63,626 |
| 训练轮数 | 10 epochs |
| 优化器 | Adam（lr = 0.01） |
| **测试集准确率** | **98.15%** |
| CPU 训练耗时 | 约 140 ~ 240 秒 |

> 上述结果为本机在 Intel CPU、无 GPU 环境下实测所得，重复运行可复现。训练耗时受机器负载影响，波动较大。
>
> 同任务的 ViT 方案（[`transformer/`](../transformer/)）实测为 97.96% 准确率、472~782 秒训练耗时、205,066 参数 —— **参数量是本方案的 3.2 倍，训练耗时约 3.4 倍，准确率反而略低**。原因分析见[主 README 的「为什么 Transformer 在这里输了」](../README.md#为什么-transformer-在这里输了)。

## 模型结构

单卷积层 + 单全连接层的轻量网络，配合 Batch Normalization 提升收敛速度：

```
输入 (1, 28, 28)
   │
   ├─ Conv2d(1 → 32, kernel_size=5, padding=2)   32 个 5×5 卷积核提取特征
   ├─ BatchNorm2d(32)                             批归一化，稳定梯度分布
   ├─ ReLU()                                      激活函数，引入非线性
   └─ MaxPool2d(2)                                最大池化，28×28 → 14×14
   │
Flatten (32 × 14 × 14 = 6,272)
   │
   └─ Linear(6,272 → 10)                         全连接层，映射到 10 个类别
   │
输出 (10)  —— logits，未经 softmax
```

### 关键设计说明

- **卷积核数量 32**：控制特征图的通道数，决定网络能从图像中提取多少种特征。
- **padding=2 保持尺寸**：`H = (28 - 5 + 2×2) / 1 + 1 = 28`，使卷积前后空间尺寸不变，便于堆叠网络层。
- **池化降维**：2×2 窗口四选一，将 28×28 压缩到 14×14，参数量下降的同时保留显著特征。
- **输出为 logits**：损失函数使用 `CrossEntropyLoss`，内部已包含 log_softmax，因此模型输出层**不接** `Softmax`，避免重复归一化。
- **权重以 `state_dict` 保存**：只存权重字典而非整个模型对象，便于跨版本迁移、切换设备，并规避 pickle 反序列化执行代码的安全风险。详见 [PyTorch 序列化建议](https://pytorch.org/docs/stable/notes/serialization.html)。
- **训练与评估模式分离**：训练循环内调用 `model.train()`、验证前调用 `model.eval()`。这对含 BatchNorm 的模型尤为关键——若评估时未切到 `eval()` 模式，BatchNorm 会继续使用当前批次的均值方差而非训练收敛的滑动统计量，导致评估结果失真。

## 文件结构

```
mnist-CNN-vs-Transformer/
├── README.md           # 仓库主 README：两方案对比与差异分析
├── requirements.txt    # 依赖声明（两方案共用，位于仓库根目录）
├── .gitignore          # 忽略规则（位于仓库根目录）
│
└── cnn/                # ← 本目录
    ├── model.py            # 模型定义：CNN 类
    ├── train_cnn.py        # 训练脚本：加载数据、训练 10 轮、每轮在测试集上评估并保存权重
    ├── visualize.py        # 可视化测试：加载权重，逐张弹窗显示图像并输出预测值
    ├── predict.py          # 自定义图片预测：对单张手写数字图片做预处理与推理
    ├── my_digit.png        # 样例测试图片（白底黑字），由 predict.py 自动生成
    └── README.md           # 本文件：CNN 方案详解
```

> 以下命令均需在本目录（`cnn/`）下执行。

运行后会自动生成以下目录（均已被 `.gitignore` 排除）：

| 路径 | 说明 |
| :--- | :--- |
| `mnist/` | 自动下载的 MNIST 原始数据，约 76 MB |
| `model/` | 训练输出的权重文件 `mnist_cnn.pth`（`state_dict` 格式，约 253 KB） |
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
> `visualize.py` 依赖 OpenCV 显示图像，若不需要可视化测试可忽略该依赖。

### 2. 训练模型

```bash
python train_cnn.py
```

首次运行会自动下载 MNIST 数据集。每轮训练结束后在测试集上评估并打印准确率，训练完成后权重保存至 `model/mnist_cnn.pth`。

### 3. 可视化测试

```bash
python visualize.py
```

加载训练好的权重，在测试集上逐张弹出图像（放大 4 倍）并打印预测值与真实值。

- 按**任意键**继续下一张，按 **`q`** 提前退出
- 需在图形界面环境下运行，无 GUI 的服务器或 Jupyter Notebook 中 `imshow` 无法工作

### 4. 识别自定义图片

```bash
python predict.py
```

用于识别自己的手写数字照片。修改脚本顶部的配置项即可：

```python
WEIGHT_PATH = "model/mnist_cnn.pth"   # 权重路径
IMAGE_PATH  = "my_digit.png"          # 待识别图片路径
INVERT      = True                    # 颜色反转开关
```

> ⚠️ **关于 `INVERT` 开关**：MNIST 的数据约定是「黑底白字」，而日常拍摄的照片、扫描件通常是「白底黑字」。
> - 图片是**白底黑字** → `INVERT = True`（需反转成黑底白字）
> - 图片已经是**黑底白字** → `INVERT = False`
>
> 该开关设错是自定义图片识别失败的最常见原因。

若 `IMAGE_PATH` 指向的图片不存在，脚本会自动从 MNIST 测试集导出一张样例图（放大至 128×128 并反转为白底黑字），保证开箱即可跑通。运行结果如下：

```text
使用设备: cpu
模型加载成功: model/mnist_cnn.pth
----------------------------------------
预测结果: 7
各类别概率:
  数字 0: 0.0000
  数字 1: 0.0000
  ...
  数字 7: 1.0000  ########################################
  ...
----------------------------------------
```

## 常见问题

**Q：数据集下载失败怎么办？**
手动下载 MNIST 并解压到项目目录下的 `mnist/MNIST/raw/`，脚本检测到数据已存在时会跳过下载。

**Q：`visualize.py` 运行时窗口不显示？**
OpenCV 的 `imshow` 需要图形界面环境，在无 GUI 的服务器或 Jupyter Notebook 中无法直接使用。这种场景下建议改用 `matplotlib` 绘图。

**Q：为什么权重保存为 `state_dict` 而不是整个模型对象？**
`torch.save(cnn, ...)` 会把**整个模型对象**（网络结构 + 权重）一并 pickle 序列化，存在两个问题：一是与 Python 类定义强耦合，改动网络结构后旧权重无法加载；二是 pickle 格式可以执行任意代码，加载不可信来源的完整模型等同于运行陌生代码。

本项目因此保存 `state_dict`（纯权重字典），加载时先实例化结构一致的模型再灌入权重：

```python
# 保存
torch.save(cnn.state_dict(), "model/mnist_cnn.pth")

# 加载
cnn = CNN()
cnn.load_state_dict(torch.load("model/mnist_cnn.pth", map_location="cpu"))
cnn.eval()
```

**Q：自定义图片识别结果总是错误？**
先检查 `INVERT` 开关是否设反。MNIST 是黑底白字，日常照片是白底黑字，`INVERT` 设错会导致模型输入分布与训练时完全不一致。次要原因是图片中的数字**没有居中或占满画布**——MNIST 的数字经过标准化处理，而随手拍的照片通常有大片留白。可尝试将数字裁剪至最小外接框后居中放入正方形画布，再做灰度与缩放。

## 参考资料

- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [torchvision MNIST 数据集说明](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)
- [PyTorch Pro 诀窍：权重保存与加载](https://pytorch.org/docs/stable/notes/serialization.html)
