import os
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
import segmentation_models_pytorch as smp
from torch.optim import Adam, SGD, AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from concurrent.futures import ThreadPoolExecutor

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
import segmentation_models_pytorch as smp
from torch.optim import Adam, SGD
# 封装 clamp 函数
def clamp_transform(x):
    return x.clamp(0, 1)

config = Config(
    device="cuda",
    root_dir="../archive/",
    train_img_dir="../archive/train_img",
    train_mask_dir="../archive/train_mask",
    test_img_dir="../archive/test_img",
    test_mask_dir="../archive/test_mask",
    valid_img_dir="../archive/valid_img",
    valid_mask_dir="../archive/valid_mask",
    backbone=smp.Unet(
        encoder_name="resnet101",
        encoder_weights='imagenet',
        in_channels=1,
        classes=1
    ),
    transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),  # Assuming grayscale images
        transforms.Lambda(clamp_transform)
    ]),
    batchsize=16,
    lr=0.0001,
    num_epochs=15,
    print_freq=1
)
train_dataset = TumorDataset(config.root_dir, config.train_img_dir, config.train_mask_dir, config.transform)
test_dataset = TumorDataset(config.root_dir, config.test_img_dir, config.test_mask_dir, config.transform)
valid_dataset = TumorDataset(config.root_dir, config.valid_img_dir, config.valid_mask_dir, config.transform)
train_loader = DataLoader(train_dataset, batch_size=config.batchsize, shuffle=True,num_workers=4,pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config.batchsize, shuffle=False,num_workers=4,pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config.batchsize, shuffle=False,num_workers=4,pin_memory=True)
from tqdm import tqdm
def dice_loss(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def plot_metrics(epoch, train_losses, valid_losses, train_ious, valid_ious):
    plt.clf()  # 清空当前图形
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
    plt.close()

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
    scheduler = ReduceLROnPlateau(
        optimizer,  # 绑定的优化器
        mode='min',  # 监控指标的模式（最小化损失）
        factor=0.5,  # 学习率衰减因子（新学习率 = 原学习率 * factor）
        patience=2,  # 等待epoch数（无改善后触发衰减）
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
            scheduler.step(avg_valid_loss)  # 根据验证集的损失调整学习率
            executor.submit(plot_metrics, epoch + 1, train_losses, valid_losses, train_ious, valid_ious)

    # 训练结束后关闭交互模式
    plt.ioff()
    plt.show()
    torch.save(model.state_dict(), f'./model/UNet_epoch_{num_epochs}.pth')


def main():
    criterion = dice_loss  # 使用自定义的 Dice Loss

    model = config.backbone
    optimizer = AdamW(
        model.parameters(),
        lr=1e-4,  # 初始学习率
        weight_decay=1e-4  # 权重衰减
    )
    train(train_loader, valid_loader, model, criterion, optimizer, num_epochs=config.num_epochs)


def load_train():
    criterion = dice_loss  # 使用自定义的 Dice Loss
    model = config.backbone
    model_path = 'model/UNet_epoch_15.pth'
    model.load_state_dict(torch.load(model_path))
    optimizer = SGD(config.backbone.parameters(), lr=1e-5, momentum=0.9,      weight_decay=1e-4,  nesterov=True)
    train(train_loader, valid_loader, model, criterion, optimizer, num_epochs=config.num_epochs)



if __name__ == '__main__':
    main()
