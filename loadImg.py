import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from U_Net_train import config

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. 加载模型
def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))  # 确保模型加载到正确设备
    model.to(device)  # 移动模型到设备
    model.eval()  # 切换到评估模式
    return model


# 2. 准备输入图像
def prepare_image(image_path, transform):
    img = Image.open(image_path).convert('L')  # 转为灰度图
    img = transform(img)  # 应用预处理转换
    img = img.unsqueeze(0)  # 添加批次维度
    return img


# 3. 执行推断
def infer_and_visualize(model, image_path, transform):
    image_name=image_path
    image_path = 'E:/1555bishe/archive/test/'+image_path
    # 准备图像
    input_image = prepare_image(image_path, transform)

    # 将图像移动到设备
    input_image = input_image.to(device)
    target_image = prepare_image(f'../archive/test_mask/{image_name}',transform)
    with torch.no_grad():
        output = model(input_image)
        output = torch.sigmoid(output)  # 应用sigmoid激活函数
        output_thresholded = (output > 0.5).float()  # 阈值处理，得到二进制掩膜

    # 可视化结果
    visualize_input_output_target(input_image[0], output_thresholded[0], target_image)


def visualize_input_output_target(input_image, output_image, target_image=None):
    # 将张量转换为numpy数组并从GPU移动到CPU
    input_image = input_image.cpu().numpy()
    output_image = output_image.cpu().numpy()

    # 可视化输入和输出
    plt.figure(figsize=(12, 6))

    # 输入图像
    plt.subplot(1, 3, 1)
    plt.title("输入图像")
    plt.imshow(input_image.squeeze(), cmap='gray')
    plt.axis('off')

    # 输出图像
    plt.subplot(1, 3, 2)
    plt.title("预测掩膜")
    plt.imshow(output_image.squeeze(), cmap='gray')
    plt.axis('off')

    # 如果目标图像存在，则显示它
    if target_image is not None:
        target_image = target_image.cpu().numpy()
        plt.subplot(1, 3, 3)
        plt.title("实际掩膜")
        plt.imshow(target_image.squeeze(), cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# 4. 主函数
def main_inference():
    plt.rcParams['font.family'] = 'SimHei'  # 用来正常显示中文标签
    model_path = 'model/resnet50_FPN_epoch_15.pth'  # 根据实际路径与文件名修改
    model = config.backbone  # 使用之前配置的模型

    # 加载模型
    model = load_model(model, model_path)

    # 输入图像路径
    image_path = "27_jpg.rf.b2a2b9811786cc32a23c46c560f04d07.jpg"  # 替换为您的图片路径

    # 进行推断并可视化
    infer_and_visualize(model, image_path, config.transform)

if __name__ == '__main__':
    main_inference()



