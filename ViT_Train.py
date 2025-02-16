
import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import timm
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
def dice_loss(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# ==================== 数据集 ====================
class TumorDataset(Dataset):
    def __init__(self, root_dir, img_dir, mask_dir, transform=None):
        self.root_dir = root_dir  # 根目录
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
        x = self.vit.patch_embed(x)
        x = x + self.vit.pos_embed
        x = self.vit.blocks(x)
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return self.decoder(x)

# ==================== 配置模型 ====================
class Config:
    def __init__(self, device, root_dir, train_img_dir, train_mask_dir, test_img_dir, test_mask_dir, valid_img_dir, valid_mask_dir, backbone, transform, batchsize, lr, num_epochs, print_freq):
        self.device = device  # cuda or cpu
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

config = Config(
    device="cuda",
    root_dir="../archive/",
    train_img_dir="../archive/train_img",
    train_mask_dir="../archive/train_mask",
    test_img_dir="../archive/test_img",
    test_mask_dir="../archive/test_mask",
    valid_img_dir="../archive/valid_img",
    valid_mask_dir="../archive/valid_mask",
    backbone=ViTSeg(
        num_classes=1,
        img_size=224
    ),
    transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),  # 假设灰度图像
        transforms.Lambda(lambda x: x.clamp(0, 1))
    ]),
    batchsize=4,
    lr=0.0001,
    num_epochs=15,
    print_freq=1
)

# 创建数据加载器
train_dataset = TumorDataset(config.root_dir, config.train_img_dir, config.train_mask_dir, config.transform)
test_dataset = TumorDataset(config.root_dir, config.test_img_dir, config.test_mask_dir, config.transform)
valid_dataset = TumorDataset(config.root_dir, config.valid_img_dir, config.valid_mask_dir, config.transform)
train_loader = DataLoader(train_dataset, batch_size=config.batchsize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.batchsize, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=config.batchsize, shuffle=False)

# ==================== 训练和验证 ====================
def calculate_iou(preds, targets):
    preds_bool = preds.bool()
    targets_bool = targets.bool()
    intersection = (preds_bool & targets_bool).float().sum((1, 2))
    union = (preds_bool | targets_bool).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

def train(train_loader, valid_loader, model, criterion, optimizer, num_epochs):
    model.to(config.device)
    train_losses = []
    train_ious = []
    valid_losses = []
    valid_ious = []
    plt.ion()
    # 替换原有的ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',  # 'min' 或 'max'，根据监测指标选择
        factor=0.5,  # 学习率降低的因子
        patience=2,  # 在多少个epoch内没有改善时触发
        threshold=0.001,
        min_lr=1e-6  # 最小学习率
    )
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
            scheduler.step(avg_valid_loss)  # 根据验证集的损失调整学习率
            executor.submit(plot_metrics, epoch + 1, train_losses, valid_losses, train_ious, valid_ious)

    # 训练结束后关闭交互模式
    plt.ioff()
    plt.show()
    torch.save(model.state_dict(), f'./model/ViT_epoch_{num_epochs}.pth')

def main():
    criterion = dice_loss  # 使用自定义的 Dice Loss
    optimizer = SGD(config.backbone.parameters(), lr=config.lr, momentum=0.9, weight_decay=0.0005)

    model = config.backbone
    train(train_loader, valid_loader, model, criterion, optimizer, num_epochs=config.num_epochs)

if __name__ == '__main__':
    main()