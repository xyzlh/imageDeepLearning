import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
import segmentation_models_pytorch as smp
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# 多头注意力模块：使用 AMP 以后，混合精度计算可加速训练
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        多头注意力模块，使用 torch.nn.MultiheadAttention 实现。
        输入形状为 (B, C, H, W)，请保持 H 和 W 尽量较小，降低内存消耗和计算量。
        """
        super(MultiHeadAttentionBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        # x 的形状: (B, C, H, W)
        B, C, H, W = x.size()
        # 将空间维度展平为序列维度: (H*W, B, C)
        x_flat = x.view(B, C, H * W).permute(2, 0, 1)
        # 使用多头注意力： query、key、value 均为 x_flat
        attn_output, _ = self.multihead_attn(x_flat, x_flat, x_flat)
        # 恢复为 (B, C, H, W)
        attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)
        return attn_output


# 带有多头注意力的 UNet 模型
class AttnUNetWithAttention(nn.Module):
    def __init__(self, encoder_name, encoder_weights, in_channels, classes, num_heads, embed_dim, dummy_input_size=224,
                 pool_size=14):
        """
        构建带有多头注意力机制的 UNet 模型。
        参数说明：
          encoder_name: UNet 使用的编码器名称。
          encoder_weights: 编码器预训练权重。
          in_channels: 输入通道数。
          classes: 输出类别数。
          num_heads: 多头注意力头数。
          embed_dim: 注意力层的嵌入维度。
          dummy_input_size: 用于推断 UNet 输出通道数的示例输入图像尺寸。
          pool_size: 注意力前进行池化的目标空间尺寸，用于降低内存和计算开销。
        """
        super(AttnUNetWithAttention, self).__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
        # 使用 dummy 输入推断 UNet 的输出通道数
        dummy = torch.zeros(1, in_channels, dummy_input_size, dummy_input_size)
        with torch.no_grad():
            unet_out = self.unet(dummy)
        out_channels = unet_out.shape[1]
        # 如果输出通道与 embed_dim 不同，使用 1x1 卷积进行映射
        if out_channels != embed_dim:
            self.channel_mapper = nn.Conv2d(out_channels, embed_dim, kernel_size=1)
        else:
            self.channel_mapper = nn.Identity()

        self.pool_size = pool_size
        self.attention = MultiHeadAttentionBlock(embed_dim, num_heads)
        self.fusion_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        # 使用 1x1 卷积将特征映射到目标类别数，保证输出维度匹配标签
        self.segmentation_head = nn.Conv2d(embed_dim, classes, kernel_size=1)

    def forward(self, x):
        # 通过 UNet 获取特征
        unet_out = self.unet(x)
        # 映射到 embed_dim
        features = self.channel_mapper(unet_out)
        # 自适应池化降低空间分辨率，减少后续注意力计算量
        pooled_features = F.adaptive_avg_pool2d(features, output_size=(self.pool_size, self.pool_size))
        # 对池化后的特征应用注意力模块
        attn_features = self.attention(pooled_features)
        # 上采样将特征恢复到原始尺寸
        attn_features_upsampled = F.interpolate(attn_features, size=features.shape[2:], mode='bilinear',
                                                align_corners=False)
        # 残差融合注意力特征与原始特征
        fused = self.fusion_conv(attn_features_upsampled) + features
        output = self.segmentation_head(fused)
        return output


# 数据集定义，适用于图像分割任务
class TumorDataset(Dataset):
    def __init__(self, root_dir, img_dir, mask_dir, transform=None):
        self.root_dir = root_dir
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


# 配置参数
class Config:
    def __init__(self, device, root_dir, train_img_dir, train_mask_dir,
                 test_img_dir, test_mask_dir, valid_img_dir, valid_mask_dir,
                 backbone, transform, batchsize, lr, num_epochs, print_freq):
        self.device = device  # cuda 或 cpu
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

# 定义 clamp_image 函数以替换 lambda，确保可以被 pickle
def clamp_image(x):
    return x.clamp(0, 1)

# 配置初始化
config = Config(
    device="cuda",
    root_dir="../archive/",
    train_img_dir="../archive/train_img",
    train_mask_dir="../archive/train_mask",
    test_img_dir="../archive/test_img",
    test_mask_dir="../archive/test_mask",
    valid_img_dir="../archive/valid_img",
    valid_mask_dir="../archive/valid_mask",
    backbone=AttnUNetWithAttention(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
        num_heads=4,
        embed_dim=256,
        dummy_input_size=224,
        pool_size=14
    ),
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),  # 针对灰度图：1通道
        transforms.Lambda(clamp_image)
    ]),
    batchsize=4,
    lr=1e-4,
    num_epochs=25,
    print_freq=1
)

# 数据加载器：增加 num_workers 和 pin_memory，加快数据加载
train_dataset = TumorDataset(config.root_dir, config.train_img_dir, config.train_mask_dir, config.transform)
test_dataset = TumorDataset(config.root_dir, config.test_img_dir, config.test_mask_dir, config.transform)
valid_dataset = TumorDataset(config.root_dir, config.valid_img_dir, config.valid_mask_dir, config.transform)

train_loader = DataLoader(train_dataset, batch_size=config.batchsize, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config.batchsize, shuffle=False, num_workers=4, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config.batchsize, shuffle=False, num_workers=4, pin_memory=True)


# 自定义 Dice Loss
def dice_loss(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


# 绘制训练和验证指标
def plot_metrics(epoch, train_losses, valid_losses, train_ious, valid_ious):
    plt.figure(figsize=(12, 5))
    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    # IoU 曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_ious) + 1), train_ious, label='Train IoU')
    plt.plot(range(1, len(valid_ious) + 1), valid_ious, label='Valid IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('IoU Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)
    plt.close()


# 计算 IoU 指标
def calculate_iou(preds, targets):
    preds_bool = preds.bool()
    targets_bool = targets.bool()
    intersection = (preds_bool & targets_bool).float().sum((1, 2))
    union = (preds_bool | targets_bool).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


# 训练流程：添加混合精度训练 (AMP) 以加快训练速度
def train(train_loader, valid_loader, model, criterion, optimizer, num_epochs):
    model.to(config.device)
    scaler = GradScaler()  # 混合精度梯度缩放器
    train_losses, train_ious = [], []
    valid_losses, valid_ious = [], []
    plt.ion()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)

    # 使用线程池异步绘图，避免阻塞训练
    with ThreadPoolExecutor(max_workers=1) as executor:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            running_iou = 0.0

            for i, (inputs, masks) in enumerate(
                    tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch", disable=False)):
                inputs = inputs.to(config.device, non_blocking=True)
                masks = masks.to(config.device, non_blocking=True)

                optimizer.zero_grad()
                with autocast():
                    outputs = model(inputs)
                    preds = torch.sigmoid(outputs)
                    loss = criterion(preds, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                binary_preds = (preds > 0.5).float()
                running_iou += calculate_iou(binary_preds.squeeze(1), masks.squeeze(1))

            epoch_loss = running_loss / len(train_loader)
            epoch_iou = running_iou / len(train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train IoU: {epoch_iou:.4f}")

            model.eval()
            valid_loss = 0.0
            valid_iou = 0.0
            with torch.no_grad():
                for inputs, masks in valid_loader:
                    inputs = inputs.to(config.device, non_blocking=True)
                    masks = masks.to(config.device, non_blocking=True)
                    with autocast():
                        outputs = model(inputs)
                        preds = torch.sigmoid(outputs)
                        loss = criterion(preds, masks)
                    valid_loss += loss.item()
                    binary_preds = (preds > 0.5).float()
                    valid_iou += calculate_iou(binary_preds.squeeze(1), masks.squeeze(1))

            avg_valid_loss = valid_loss / len(valid_loader)
            avg_valid_iou = valid_iou / len(valid_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Valid Loss: {avg_valid_loss:.4f}, Valid IoU: {avg_valid_iou:.4f}")

            train_losses.append(epoch_loss)
            train_ious.append(epoch_iou)
            valid_losses.append(avg_valid_loss)
            valid_ious.append(avg_valid_iou)
            scheduler.step(avg_valid_loss)
            executor.submit(plot_metrics, epoch + 1, train_losses, valid_losses, train_ious, valid_ious)

    plt.ioff()
    plt.show()

    torch.save(model.state_dict(), f'./model/Atten_UNet_epoch_{num_epochs}.pth')

def load_train():
    criterion = dice_loss  # 使用自定义的 Dice Loss
    model = config.backbone
    model_path = 'model/UNet_epoch_25.pth'  # 替换为你的模型路径
    model.load_state_dict(torch.load(model_path))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,  # 初始学习率调整为更小值
        weight_decay=1e-4  # 更合适的权重衰减
    )
    train(train_loader, valid_loader, model, criterion, optimizer, num_epochs=config.num_epochs)
    epo=config.num_epochs+25
    os.rename('./model/UNet_epoch_25.pth', f'./model/UNet_epoch_{epo}.pth')

def main():
    criterion = dice_loss  # 使用自定义的 Dice Loss
    model = config.backbone
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,  # 初始学习率调整为更小值
        weight_decay=1e-4  # 更合适的权重衰减
    )
    train(train_loader, valid_loader, model, criterion, optimizer, num_epochs=config.num_epochs)


if __name__ == '__main__':
    main()
