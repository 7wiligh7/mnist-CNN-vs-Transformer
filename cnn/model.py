import torch


class CNN(torch.nn.Module):
    """
    轻量级卷积神经网络，用于 MNIST 手写数字分类。

    网络结构：Conv -> BatchNorm -> ReLU -> MaxPool -> Flatten -> Linear
    """

    def __init__(self):
        super(CNN, self).__init__()
        self.conv = torch.nn.Sequential(
            # 卷积层：1 个输入通道 -> 32 个特征图
            torch.nn.Conv2d(in_channels=1,   # 输入通道数（灰度图为 1）
                            out_channels=32, # 卷积核数量，即输出特征图数量
                            kernel_size=5,   # 卷积核大小 5x5
                            # padding=2 保持空间尺寸不变：H = (28 - 5 + 2*2)/1 + 1 = 28
                            padding=2),
            # 批归一化：稳定梯度分布，加快收敛
            torch.nn.BatchNorm2d(32),
            # 激活函数：引入非线性
            torch.nn.ReLU(),
            # 最大池化：2x2 窗口四选一，28x28 -> 14x14
            torch.nn.MaxPool2d(2),
        )
        # 全连接层：展平后 32*14*14 -> 10 类
        self.fc = torch.nn.Linear(in_features=32 * 14 * 14,
                                  out_features=10)

    def forward(self, x):
        out = self.conv(x)
        # 将 (N, C, H, W) 的特征图展平为 (N, C*H*W) 的一维张量
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out
