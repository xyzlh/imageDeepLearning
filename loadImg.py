import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from U_Net_train import  config,TumorDataset  # 假设U_Net_train.py包含数据集和配置
import segmentation_models_pytorch as smp
from torch.optim import Adam
from U_Net_train import  train_loader,test_loader,valid_dataset

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 计算IoU


def calculate_iou(preds, targets):
    # 确保输入为二值张量（0或1）
    preds_bool = preds.bool()  # 若 preds 是浮点型（0.0/1.0），直接转布尔型
    targets_bool = targets.bool()

    intersection = (preds_bool & targets_bool).float().sum((1, 2))  # 按批次计算交集
    union = (preds_bool | targets_bool).float().sum((1, 2))  # 按批次计算并集
    iou = (intersection + 1e-6) / (union + 1e-6)  # 防止除以零
    return iou.mean().item()  # 返回平均IoU

# 在验证集上评估模型
def evaluate_model(model, valid_loader):
    model.eval()
    total_iou = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 推理
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            preds = (outputs > 0.5).float()  # 阈值处理

            # 计算IoU
            batch_iou = calculate_iou(preds.squeeze(1), targets.squeeze(1))
            total_iou += batch_iou * inputs.size(0)
            total_samples += inputs.size(0)

    mean_iou = total_iou / total_samples
    return mean_iou


# 主函数
def main():
    # 加载模型
    model = config.backbone.to(device)
    model_path = 'model/UNet_epoch_1.pth'  # 替换为你的模型路径
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 加载验证集
    valid_loader =train_loader

    # 评估模型
    mean_iou = evaluate_model(model, valid_loader)
    print(f"验证集平均IoU: {mean_iou:.4f}")


if __name__ == '__main__':
    main()