"""
评估 Vision Transformer 在测试集上的分类准确率。

用法：python evaluate.py
"""
import torch

from data_loader import loader
from model import VisionTransformer

WEIGHT_PATH = "vit_transformer.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate():
    _, test_loader = loader()

    model = VisionTransformer().to(device)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    model.eval()  # 切换为评估模式（禁用 Dropout 等）

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print("测试集准确率：{:.2f}%".format(accuracy))


if __name__ == "__main__":
    evaluate()
