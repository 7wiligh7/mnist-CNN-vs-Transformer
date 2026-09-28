"""
使用训练好的 CNN 模型识别单张手写数字图片。

与训练时的预处理保持一致：灰度化 -> 颜色反转 -> 缩放到 28x28 -> 归一化。

注意：MNIST 的数据约定是「黑底白字」，而日常拍摄的照片通常是「白底黑字」，
因此需要通过 INVERT 开关决定是否做颜色反转，否则预测结果会明显变差。

用法：python predict.py
"""
import os

import torch
import torchvision.transforms as transforms
from PIL import Image

from model import CNN

# ==================== 1. 配置项 ====================
# 模型权重路径（由 train_cnn.py 训练后生成）
WEIGHT_PATH = "model/mnist_cnn.pth"

# 待识别的图片路径，可自行修改
IMAGE_PATH = "my_digit.png"

# 颜色反转开关
#   True  —— 输入图片是「白底黑字」（如照片、扫描件），需反转成 MNIST 的「黑底白字」
#   False —— 输入图片已经是「黑底白字」，无需反转
INVERT = True


# ==================== 2. 图片预处理 ====================
def preprocess_image(image_path):
    """
    将单张图片转换为模型可接受的张量。

    步骤：灰度化 -> 颜色反转 -> 缩放至 28x28 -> ToTensor + Normalize -> 增加 batch 维

    返回：
        Tensor，形状 (1, 1, 28, 28)
    """
    # 读取图片并转换为灰度模式 'L'，确保通道数为 1
    img = Image.open(image_path).convert("L")

    # 颜色反转：255 - x，黑底白字与白底黑字互转
    if INVERT:
        img = Image.eval(img, lambda x: 255 - x)

    # 缩放至 MNIST 的 28x28 尺寸
    # LANCZOS 为高质量插值算法，比默认的双线性插值保留更多笔画细节
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # 与训练时完全一致的预处理管线
    transform = transforms.Compose([
        transforms.ToTensor(),                        # PIL/numpy -> Tensor，像素值归一化到 [0, 1]
        transforms.Normalize((0.1307,), (0.3081,)),   # (x - mean) / std，使用 MNIST 全局均值与标准差
    ])

    # unsqueeze(0) 增加 batch 维：(1, 28, 28) -> (1, 1, 28, 28)
    return transform(img).unsqueeze(0)


# ==================== 3. 加载模型 ====================
def load_model(weight_path, device):
    """
    重建网络结构并加载权重字典。

    权重以 state_dict 形式保存，因此必须先实例化一个结构完全一致的模型，
    再把权重灌入其中。
    """
    model = CNN()
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # 切换到评估模式
    return model


# ==================== 4. 预测 ====================
def predict(model, image_tensor, device):
    """
    执行推理，返回预测数字与各类别的概率分布。
    """
    image_tensor = image_tensor.to(device)
    with torch.no_grad():  # 推理阶段不记录梯度，节省显存
        outputs = model(image_tensor)         # (1, 10) 的 logits
        prob = torch.softmax(outputs, dim=1)  # 手动转概率（模型输出层未接 Softmax）
        predicted = prob.argmax(dim=1)        # 取概率最大的类别
    return predicted.item(), prob[0].cpu().numpy()


# ==================== 5. 样例图片生成 ====================
def export_sample_image(save_path, index=0):
    """
    当用户还没有自己的图片时，从 MNIST 测试集中导出一张样例图，
    并模拟日常照片的「白底黑字」效果，保证脚本开箱即可跑通。
    """
    import torchvision.datasets as dataset

    test_data = dataset.MNIST(root="mnist", train=False, download=True)
    image, label = test_data[index]  # PIL Image，28x28，「黑底白字」

    # 放大到 128x128，模拟真实拍摄图片的尺寸
    image = image.resize((128, 128), Image.Resampling.LANCZOS)
    # 反转为「白底黑字」，与日常照片一致
    image = Image.eval(image, lambda x: 255 - x)
    image.save(save_path)
    return label


# ==================== 6. 主程序 ====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备: {}".format(device))

    # 若图片不存在，自动导出一张样例图，保证脚本可以直接运行
    if not os.path.exists(IMAGE_PATH):
        print("未找到图片 {}，正在从 MNIST 测试集导出样例图...".format(IMAGE_PATH))
        true_label = export_sample_image(IMAGE_PATH)
        print("样例图已导出，该图片的真实数字为 {}".format(true_label))

    model = load_model(WEIGHT_PATH, device)
    print("模型加载成功: {}".format(WEIGHT_PATH))

    img_tensor = preprocess_image(IMAGE_PATH)
    digit, probs = predict(model, img_tensor, device)

    print("-" * 40)
    print("预测结果: {}".format(digit))
    print("各类别概率:")
    for i, p in enumerate(probs):
        print("  数字 {}: {:.4f}  {}".format(i, p, "#" * int(p * 40)))
    print("-" * 40)
