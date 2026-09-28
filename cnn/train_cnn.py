"""
训练 CNN 模型并在测试集上评估。

用法：python train_cnn.py
"""
import os

import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils

from model import CNN

# 数据集与权重输出路径
DATA_ROOT = "mnist"
WEIGHT_DIR = "model"
WEIGHT_PATH = os.path.join(WEIGHT_DIR, "mnist_cnn.pth")

# ==================== 1. 加载数据 ====================
train_data = dataset.MNIST(
    root=DATA_ROOT,
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)

test_data = dataset.MNIST(
    root=DATA_ROOT,
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)

# ==================== 2. 分批加载 ====================
train_loader = data_utils.DataLoader(dataset=train_data,
                                     batch_size=64,
                                     shuffle=True)

test_loader = data_utils.DataLoader(dataset=test_data,
                                    batch_size=64,
                                    # 测试集不需要打乱，shuffle 必须为 False
                                    shuffle=False)

# ==================== 3. 初始化模型 ====================
cnn = CNN()

# 损失函数：CrossEntropyLoss 内部已含 log_softmax，
# 因此模型输出层不再接 Softmax，避免重复归一化
loss_func = torch.nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)

# ==================== 4. 训练循环 ====================
# epoch = 一次完整遍历训练集
for epoch in range(10):
    cnn.train()
    for images, labels in train_loader:
        # 前向传播
        outputs = cnn(images)
        # 计算损失：传入输出层 logits 与真实标签
        loss = loss_func(outputs, labels)

        # 清空梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

    # ==================== 5. 测试集验证 ====================
    cnn.eval()
    with torch.no_grad():
        correct = 0
        for images, labels in test_loader:
            outputs = cnn(images)
            # 预测类别：取 logits 最大的下标
            _, pred = outputs.max(1)
            # eq() 逐元素比较，相等为 True，累加得到正确预测数
            correct += (pred == labels).sum().item()

    print("Epoch [{}/10] 测试集准确率: {:.2%}".format(
        epoch + 1, correct / len(test_data)))

# ==================== 6. 保存权重 ====================
# 使用 state_dict（纯权重字典）而非整个模型对象进行序列化：
# 1. 体积更小，不与 Python 类定义耦合，可跨版本 / 跨代码结构加载
# 2. pickle 格式可执行任意代码，加载不可信来源的完整模型存在安全风险
# 详见 https://pytorch.org/docs/stable/notes/serialization.html
os.makedirs(WEIGHT_DIR, exist_ok=True)  # 首次运行时目录不存在，需先创建
torch.save(cnn.state_dict(), WEIGHT_PATH)
print("模型已保存至 {}".format(WEIGHT_PATH))
