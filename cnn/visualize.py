"""
可视化测试：加载训练好的 CNN，在测试集上逐张弹出图像并输出预测值。

依赖 OpenCV 的图形界面，无 GUI 环境（服务器 / Jupyter Notebook）中无法运行。

用法：python visualize.py
"""
import cv2
import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils

from model import CNN

DATA_ROOT = "mnist"
WEIGHT_PATH = "model/mnist_cnn.pth"

test_data = dataset.MNIST(
    root=DATA_ROOT,
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)

test_loader = data_utils.DataLoader(dataset=test_data,
                                    batch_size=64,
                                    shuffle=False)

# ==================== 加载模型 ====================
# 权重以 state_dict 形式保存，因此先实例化结构一致的模型再灌入权重
cnn = CNN()
cnn.load_state_dict(torch.load(WEIGHT_PATH, map_location="cpu"))
cnn.eval()  # 切换到评估模式，关闭 dropout 等训练专用行为

# ==================== 评估 ====================
loss_func = torch.nn.CrossEntropyLoss()
total_loss = 0.0
correct = 0
batch_count = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = cnn(images)
        _, pred = outputs.max(1)

        # loss_func 默认对整个 batch 取平均，需乘以 batch_size 还原为总和
        total_loss += loss_func(outputs, labels).item() * images.size(0)
        correct += (pred == labels).sum().item()
        batch_count += 1

        # 转为 numpy 便于 OpenCV 显示
        images = images.cpu().numpy()
        labels = labels.cpu().numpy()
        pred = pred.cpu().numpy()

        for idx in range(images.shape[0]):
            # images 形状为 (N, C, H, W)，单张取 (C, H, W) = (1, 28, 28)
            # 先转成 (H, W, C) = (28, 28, 1)，再 squeeze 掉单通道维，
            # 得到 OpenCV 需要的 (28, 28)
            im_data = images[idx].transpose(1, 2, 0).squeeze()
            # ToTensor 输出的是 float32 且值域为 [0, 1]，
            # OpenCV 只接受 8 位整型，不做转换会显示为全黑
            im_data = (im_data * 255).astype("uint8")

            print("预测值为{}，真实值为{}".format(pred[idx], labels[idx]))

            # 放大 4 倍便于肉眼观察（28x28 窗口过小）
            im_show = cv2.resize(im_data, (112, 112), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("当前图像（按任意键继续，按 q 退出）", im_show)

            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                print("已提前退出")
                raise SystemExit(0)

print("-" * 40)
print("平均损失: {:.4f}".format(total_loss / len(test_data)))
print("测试集准确率: {:.2%}".format(correct / len(test_data)))
print("总批次: {}".format(batch_count))

cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口，避免残留
