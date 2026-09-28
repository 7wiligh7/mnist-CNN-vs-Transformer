"""
训练 Vision Transformer 模型。

用法：python train.py
"""
import os

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import loader
from model import VisionTransformer

# 训练轮数
EPOCHS = 10

# 权重输出路径
WEIGHT_PATH = "vit_transformer.pth"

# ==================== 1. 设备配置 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== 2. 训练 ====================
def train():
    train_loader, _ = loader()

    model = VisionTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Adam 优化器，学习率 0.001
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数（自带 softmax）

    print("开始训练，共 {} 轮，设备：{}".format(EPOCHS, device))

    for epoch in range(EPOCHS):
        model.train()  # 设置为训练模式（启用 Dropout 等）
        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 标准训练步骤
            optimizer.zero_grad()  # 清空梯度
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            total_loss += loss.item()

        print("Epoch [{}/{}] 平均Loss: {:.4f}".format(
            epoch + 1, EPOCHS, total_loss / len(train_loader)))

    # ==================== 3. 保存权重 ====================
    # 使用 state_dict（纯权重字典）而非整个模型对象进行序列化
    torch.save(model.state_dict(), WEIGHT_PATH)
    print("模型已保存至 {}".format(WEIGHT_PATH))
    print("提示：运行 evaluate.py 查看测试集准确率")


if __name__ == "__main__":
    train()
