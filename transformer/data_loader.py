import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

def loader():
    # ==================== 1. 数据加载与预处理 ====================
    # 定义数据预处理的管道组合
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将PIL图像或numpy数组转换为tensor，并将像素值从[0,255]归一化到[0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化：(x - mean) / std
        # 0.1307是MNIST数据集的全局均值，0.3081是全局标准差
    ])
    # 下载并加载训练数据集
    train_dataset = torchvision.datasets.MNIST(
        root='./mnist',  # 数据存储路径
        train=True,  # True表示加载训练集，False表示测试集
        download=True,  # 如果本地没有数据，则自动下载
        transform=transform) # 应用上面定义的预处理

    # 下载并加载测试数据集
    test_dataset = torchvision.datasets.MNIST(
        root='./mnist',
        train=False,
        download=True,
        transform=transform)
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,  # 每批次加载64张图片
        shuffle=True  # 每个epoch打乱数据顺序，增加训练随机性
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False  # 测试时不需要打乱
    )
    return train_loader, test_loader