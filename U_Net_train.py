import numpy as np
import pandas as pd
import os
import json
import pprint
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import matplotlib.patches as patches
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch

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
from torch.optim import Adam

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
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=1,
        classes=1
    ),
    transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),  # Assuming grayscale images
        transforms.Lambda(lambda x: x.clamp(0, 1))
    ]),
    batchsize=4,
    lr=0.001,
    num_epochs=1,
    print_freq=1
)
train_dataset = TumorDataset(config.root_dir, config.train_img_dir, config.train_mask_dir, config.transform)
test_dataset = TumorDataset(config.root_dir, config.test_img_dir, config.test_mask_dir, config.transform)
valid_dataset = TumorDataset(config.root_dir, config.valid_img_dir, config.valid_mask_dir, config.transform)
train_loader = DataLoader(train_dataset, batch_size=config.batchsize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.batchsize, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=config.batchsize, shuffle=False)
from tqdm import tqdm


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

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_iou = 0.0

        # 使用 tqdm 包装 train_loader，以显示进度条
        for i, (inputs, masks) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch", disable=False)):

            inputs = inputs.to(config.device)
            masks = masks.to(config.device)

            outputs = model(inputs)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 修改训练和验证部分的预测逻辑
            preds = (torch.sigmoid(outputs) > 0.5).float()  # 二值化预测结果
            iou = calculate_iou(preds.squeeze(1), masks.squeeze(1))  # 统一维度
            running_iou += iou

        # 在每个 epoch 结束后，计算平均损失和 mIoU
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_miou = running_iou / len(train_loader)  # 平均 IoU

        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {epoch_loss:.4f}, Mean IoU: {epoch_miou:.4f}")

        model.eval()
        valid_loss = 0.0
        valid_iou = 0.0

        with torch.no_grad():
            for inputs, masks in valid_loader:
                inputs = inputs.to(config.device)
                masks = masks.to(config.device)

                outputs = model(inputs)
                loss = criterion(outputs, masks)
                valid_loss += loss.item()

                # 计算验证集的预测结果
                preds = (torch.sigmoid(outputs) > 0.5).float()  # 二值化预测结果
                iou = calculate_iou(preds.squeeze(1), masks.squeeze(1))  # 统一维度
                valid_iou += iou

        avg_valid_loss = valid_loss / len(valid_loader)
        avg_valid_miou = valid_iou / len(valid_loader)  # 平均验证 IoU

        print(f'Epoch [{epoch + 1}/{num_epochs}] Average Validation Loss: {avg_valid_loss:.4f}, Average Validation Mean IoU: {avg_valid_miou:.4f}')

    torch.save(model.state_dict(), f'./model/UNet_epoch_{num_epochs}.pth')

def main():
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = Adam(config.backbone.parameters(), lr=config.lr)
    model = config.backbone
    train(train_loader, valid_loader, model, criterion, optimizer, num_epochs=config.num_epochs)


if __name__ == '__main__':
    main()
