import os
from PIL import Image
import matplotlib.pyplot as plt
from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3PlusDecoder
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
import segmentation_models_pytorch as smp
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from concurrent.futures import ThreadPoolExecutor
from loadImg import device
import segmentation_models_pytorch as smp
from torch.optim import Adam, SGD
from tqdm import tqdm
# 封装 clamp 函数
def clamp_transform(x):
    return x.clamp(0, 1)

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


class DynamicChannelAttention(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        # 动态调整压缩比（来自文献[5]）
        self.reduction = nn.Parameter(torch.tensor(reduction, dtype=torch.float))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel // self.reduction)),
            nn.GELU(),  # 替换ReLU
            nn.Linear(int(channel // self.reduction), channel),
            nn.Hardsigmoid()  # 比Sigmoid更适合医学图像[4](@ref)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_y = self.avg_pool(x).view(b, c)
        max_y = self.max_pool(x).view(b, c)
        y = self.fc(avg_y + max_y).view(b, c, 1, 1)
        return x * y


class LightSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # 深度可分离卷积（来自文献[6]）
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        return x * self.conv(y)


class EnhancedDualASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 原始ASPP
        self.aspp = smp.base.modules.ASPP(
            in_channels=in_channels,
            out_channels=out_channels,
            dilations=[1, 6, 12, 18]  # 增加dilation=1的卷积[4](@ref)
        )

        # 动态注意力分支（文献[5]建议）
        self.ch_att = DynamicChannelAttention(out_channels)  # 作用于ASPP输出
        self.sp_att = LightSpatialAttention()

        # 残差连接系数（文献[6]）
        self.alpha = nn.Parameter(torch.tensor(0.3))
        self.beta = nn.Parameter(torch.tensor(0.7))

    def forward(self, x):
        aspp_out = self.aspp(x)
        ch_att = self.ch_att(aspp_out)
        sp_att = self.sp_att(aspp_out)
        # 动态混合公式（来自文献[5]）
        return self.alpha * ch_att + self.beta * sp_att
config = Config(
    device="cuda",
    root_dir="../archive/",
    train_img_dir="../archive/train_img",
    train_mask_dir="../archive/train_mask",
    test_img_dir="../archive/test_img",
    test_mask_dir="../archive/test_mask",
    valid_img_dir="../archive/valid_img",
    valid_mask_dir="../archive/valid_mask",
    backbone=smp.DeepLabV3Plus(  # 使用自定义模型
        encoder_name="resnet50",
        encoder_weights='imagenet',
        in_channels=1,
        classes=1,
        decoder_aspp_class=EnhancedDualASPP  # 替换原始ASPP
    ),
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),  # Assuming grayscale images
        transforms.Lambda(clamp_transform)
    ]),
    batchsize=16,
    lr=3e-5,
    num_epochs=25,
    print_freq=1
)
train_dataset = TumorDataset(config.root_dir, config.train_img_dir, config.train_mask_dir, config.transform)
valid_dataset = TumorDataset(config.root_dir, config.valid_img_dir, config.valid_mask_dir, config.transform)
train_loader = DataLoader(train_dataset, batch_size=config.batchsize, shuffle=True ,num_workers=4,pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config.batchsize, shuffle=False,num_workers=4,pin_memory=True)

def dice_loss(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


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
    # 替换原有的ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimizer,  # 绑定的优化器
        mode='min',  # 监控指标的模式（最小化损失）
        factor=0.2,  # 学习率衰减因子（新学习率 = 原学习率 * factor）
        patience=3,  # 等待epoch数（无改善后触发衰减）
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        scaler = torch.amp.GradScaler('cuda')
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            running_iou = 0.0

            for i, (inputs, masks) in enumerate(
                    tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch", disable=False)):
                inputs = inputs.to(config.device)
                masks = masks.to(config.device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    preds = torch.sigmoid(outputs)
                    loss = criterion(preds, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
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
            scheduler.step(avg_valid_loss)
            executor.submit(plot_metrics, epoch + 1, train_losses, valid_losses, train_ious, valid_ious)

    # 训练结束后关闭交互模式
    plt.ioff()
    plt.show()
    torch.save(model.state_dict(), f'./model/atten_DeepLabV3plus_epoch_{num_epochs}.pth')

def load_train():
    criterion = dice_loss  # 使用自定义的 Dice Loss
    model = config.backbone

    model_path = 'model/DeepLabV3plus_epoch_15.pth'
    model.load_state_dict(torch.load(model_path),strict=False)
    optimizer = SGD(config.backbone.parameters(), lr=1e-5, momentum=0.9,      weight_decay=1e-4,  nesterov=True)
    train(train_loader, valid_loader, model, criterion, optimizer, num_epochs=config.num_epochs)


def main():
    criterion = dice_loss  # 使用自定义的 Dice Loss
    model = config.backbone

    optimizer = AdamW(
        model.parameters(),
        lr=1e-4,  # 初始学习率
        weight_decay=1e-4  # 权重衰减
    )
    train(train_loader, valid_loader, model, criterion, optimizer, num_epochs=config.num_epochs)


if __name__ == '__main__':
    main()
