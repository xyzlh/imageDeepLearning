from concurrent.futures import ThreadPoolExecutor

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
def plot_metrics(epoch, train_losses, valid_losses, train_ious, valid_ious):
    # 创建两个并排的子图
    plt.figure(figsize=(12, 5))

    # Loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1),train_losses, label='Train Loss')
    plt.plot(range(1, len(valid_losses) + 1),valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)

    # IoU曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_ious) + 1),train_ious, label='Train IoU')
    plt.plot(range(1, len(valid_ious) + 1),valid_ious, label='Valid IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('IoU Curve')
    plt.legend()
    plt.grid(True)

    # 调整布局并显示
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)  # 短暂暂停让图像更新
    plt.close()  # 关闭当前图像，避免内存泄漏
# 修改 Config 类的 backbone 定义
class Config:
    def __init__(self, device, root_dir, train_img_dir, train_mask_dir,
                 test_img_dir, test_mask_dir, valid_img_dir, valid_mask_dir,
                 backbone, transform, batchsize, lr, num_epochs, print_freq):
        self.device = device  # cuda or 0 or cpu
        self.root_dir = root_dir
        self.train_img_dir = train_img_dir
        self.train_mask_dir = train_mask_dir
        self.test_img_dir = test_img_dir
        self.test_mask_dir = test_mask_dir
        self.valid_img_dir = valid_img_dir
        self.valid_mask_dir = valid_mask_dir
        self.backbone = backbone
        self.transform = transform
        self.batchsize = batchsize
        self.lr = lr
        self.num_epochs = num_epochs
        self.print_freq = print_freq




class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, in_channels, num_heads=4, reduction_ratio=4):
        super().__init__()
        assert in_channels % num_heads == 0, "输入通道数必须能被头数整除"

        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        # 通道注意力（保持原设计）
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 空间注意力（调整为多头兼容）
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.Conv2d(in_channels // reduction_ratio, num_heads, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # 通道注意力
        channel_att = self.channel_att(x)

        # 空间注意力
        spatial_att = self.spatial_att(x)  # [B, num_heads, H, W]
        spatial_att = spatial_att.unsqueeze(2)  # [B, num_heads, 1, H, W]

        # 分解到多头
        x = x.view(B, self.num_heads, self.head_dim, H, W)  # [B, H, C/H, H, W]

        # 应用注意力
        attended = x * spatial_att
        attended = attended.view(B, C, H, W)

        return attended * channel_att + x  # 残差连接
class CustomDeepLabV3Plus(smp.DeepLabV3Plus):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 在ASPP后添加多头注意力
        self.aspp_attention = MultiHeadAttentionBlock(
            in_channels=256,  # 匹配ASPP输出通道数
            num_heads=4
        )


        # 替换原始解码器
        self._decoder = nn.Sequential(
            MultiHeadAttentionBlock(in_channels=256 + 48),
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # 保持原始解码器参数
        self.conv1 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        def forward(self, x):
            """修改后的前向传播逻辑"""
            # 获取编码器特征
            features = self.encoder(x)

            # ASPP处理
            aspp_output = self.aspp(features[-1])
            aspp_output = self.aspp_attention(aspp_output)  # 添加注意力

            # 解码器特征融合
            low_level_features = features[0]  # 来自ResNet的layer1
            decoder_output = self.decoder(aspp_output, low_level_features)

            # 上采样和输出
            masks = self.segmentation_head(decoder_output)
            if self.upsampling == 8:
                masks = F.interpolate(masks, scale_factor=8, mode='bilinear', align_corners=True)
            return masks

        @property
        def decoder(self):
            """动态返回解码器组件"""
            return self._decoder

        @decoder.setter
        def decoder(self, value):
            """保留原始解码器接口"""
            self._decoder = value
config = Config(
    device="cuda",
    root_dir="../archive/",
    train_img_dir="../archive/train_img",
    train_mask_dir="../archive/train_mask",
    test_img_dir="../archive/test_img",
    test_mask_dir="../archive/test_mask",
    valid_img_dir="../archive/valid_img",
    valid_mask_dir="../archive/valid_mask",
    backbone=CustomDeepLabV3Plus(  # 改为 DeepLabV3+
        encoder_name="resnet50",
        encoder_weights="imagenet",  # 启用预训练权重
        in_channels=1,  # 输入为单通道
        classes=1  # 输出单通道
    ),

    transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),  # Assuming grayscale images
        transforms.Lambda(lambda x: x.clamp(0, 1))
    ]),
    batchsize=4,
    lr=0.001,
    num_epochs=5,
    print_freq=1
)


# 修改数据集类：确保输入为三通道（如果使用预训练权重）
class TumorDataset(Dataset):
    def __init__(self, root_dir, img_dir, mask_dir, transform=None):
        self.root_dir = root_dir  # /kaggle/working/
        self.transform = transform
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_files = os.listdir(self.img_dir)
        self.mask_files = os.listdir(self.mask_dir)

    def __len__(self):
        return len(self.mask_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img_gray = img.convert('L')

        mask_name = self.mask_files[idx]
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            img_gray = self.transform(img_gray)
            mask = self.transform(mask)

        return img_gray, mask

# 其余代码保持不变（train、calculate_iou 等函数）

train_dataset = TumorDataset(config.root_dir, config.train_img_dir, config.train_mask_dir, config.transform)
test_dataset = TumorDataset(config.root_dir, config.test_img_dir, config.test_mask_dir, config.transform)
valid_dataset = TumorDataset(config.root_dir, config.valid_img_dir, config.valid_mask_dir, config.transform)
train_loader = DataLoader(train_dataset, batch_size=config.batchsize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.batchsize, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=config.batchsize, shuffle=False)
def dice_loss(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def calculate_iou(preds, targets):
    # 确保输入为二值张量（0或1）
    preds_bool = preds.bool()  # 若 preds 是浮点型（0.0/1.0），直接转布尔型
    targets_bool = targets.bool()

    intersection = (preds_bool & targets_bool).float().sum((1, 2))  # 按批次计算交集
    union = (preds_bool | targets_bool).float().sum((1, 2))  # 按批次计算并集
    iou = (intersection + 1e-6) / (union + 1e-6)  # 防止除以零
    return iou.mean().item()  # 返回平均IoU


def train(train_loader, valid_loader, model, criterion, optimizer, num_epochs):
    model.to(config.device)
    train_losses = []
    train_ious = []
    valid_losses = []
    valid_ious = []
    plt.ion()

    with ThreadPoolExecutor(max_workers=1) as executor:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            running_iou = 0.0

            for i, (inputs, masks) in enumerate(
                    tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch", disable=False)):
                inputs = inputs.to(config.device)
                masks = masks.to(config.device)

                outputs = model(inputs)

                # 使用 sigmoid 将输出转换为概率
                preds = torch.sigmoid(outputs)

                # 计算 Dice Loss
                loss = criterion(preds, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # 二值化预测结果
                binary_preds = (preds > 0.5).float()
                iou = calculate_iou(binary_preds.squeeze(1), masks.squeeze(1))
                running_iou += iou

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_miou = running_iou / len(train_loader)

            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {epoch_loss:.4f}, Mean IoU: {epoch_miou:.4f}")

            # 验证阶段
            model.eval()
            valid_loss = 0.0
            valid_iou = 0.0

            with torch.no_grad():
                for inputs, masks in valid_loader:
                    inputs = inputs.to(config.device)
                    masks = masks.to(config.device)

                    outputs = model(inputs)

                    preds = torch.sigmoid(outputs)
                    loss = criterion(preds, masks)
                    valid_loss += loss.item()

                    binary_preds = (preds > 0.5).float()
                    iou = calculate_iou(binary_preds.squeeze(1), masks.squeeze(1))
                    valid_iou += iou

            avg_valid_loss = valid_loss / len(valid_loader)
            avg_valid_miou = valid_iou / len(valid_loader)

            print(
                f'Epoch [{epoch + 1}/{num_epochs}] Average Validation Loss: {avg_valid_loss:.4f}, Average Validation Mean IoU: {avg_valid_miou:.4f}')

            train_losses.append(epoch_loss)
            train_ious.append(epoch_miou)
            valid_losses.append(avg_valid_loss)
            valid_ious.append(avg_valid_miou)

            executor.submit(plot_metrics, epoch + 1, train_losses, valid_losses, train_ious, valid_ious)

    plt.ioff()
    plt.show()
    torch.save(model.state_dict(), f'./model/DeepLabV3plus_epoch_{num_epochs}.pth')


def main():
    criterion = dice_loss  # 使用自定义的 Dice Loss
    optimizer = Adam(config.backbone.parameters(), lr=config.lr)
    model = config.backbone

    train(train_loader, valid_loader, model, criterion, optimizer, num_epochs=config.num_epochs)

if __name__ == '__main__':
    main()