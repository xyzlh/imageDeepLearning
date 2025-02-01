import os
import torch
import timm
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np


# ==================== ViT分割模型 ====================
class ViTSeg(nn.Module):
    def __init__(self, num_classes=1, img_size=224):
        super().__init__()
        self.vit = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=True,
            in_chans=1,
            img_size=img_size
        )
        self._adapt_first_conv()
        del self.vit.cls_token
        self.vit.pos_embed = nn.Parameter(self.vit.pos_embed[:, 1:, :])
        self._freeze_layers(6)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(192, 128, 4, 4),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ConvTranspose2d(64, num_classes, 4, 4)
        )

    def _adapt_first_conv(self):
        orig_conv = self.vit.patch_embed.proj
        new_conv = nn.Conv2d(
            1,
            orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding
        )
        with torch.no_grad():
            new_conv.weight.data = orig_conv.weight.data.mean(dim=1, keepdim=True)
            new_conv.bias.data = orig_conv.bias.data.clone()

        self.vit.patch_embed.proj = new_conv

    def _freeze_layers(self, num_layers):
        for i, block in enumerate(self.vit.blocks):
            if i < num_layers:
                for param in block.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.vit.patch_embed(x)  # [B, N, C]
        x = x + self.vit.pos_embed
        x = self.vit.blocks(x)
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return self.decoder(x)


# ==================== 加载模型并进行推理 ====================
def load_model(model_path):
    model = ViTSeg(num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式
    return model


def predict_and_plot(model, image_path, transform):
    img = Image.open(image_path).convert('L')  # 读取单通道灰度图
    img_transformed = transform(img).unsqueeze(0).to(device)  # 转换并维度扩展

    with torch.no_grad():
        output = model(img_transformed)  # 进行预测
        pred_mask = (torch.sigmoid(output) > 0.5).float().cpu().numpy()[0, 0]  # 二值化处理

    # 绘制结果
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(np.array(img), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Ground Truth Mask")  # 假设你也有真实掩码可以传入
    truth_mask = img_transformed.squeeze().cpu().numpy()  # 用真实掩码替代
    plt.imshow(truth_mask, cmap='gray')
    plt.axis('off')

    plt.show()


# ==================== 主程序 ====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 配置路径与参数
    model_path = './model/ViT_epoch_3.pth'  # 替换为你的模型路径
    image_path = 'E:/1555bishe/archive/train_img/2_jpg.rf.fded76c07e967829600f3509288fdfe0.jpg'  # 替换为你的测试图像路径

    # 定义转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 根据模型训练时的标准化参数进行设置
    ])

    # 加载模型
    model = load_model(model_path)

    # 进行预测并绘制结果
    predict_and_plot(model, image_path, transform)
