import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np

from U_Net_train import TumorDataset, config

train_dataset = TumorDataset(config.root_dir, config.train_img_dir, config.train_mask_dir,transform= transforms.ToTensor())

dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False)

# 初始化变量
mean = 0.
std = 0.
total_images = 0.

for images, _ in dataloader:
    batch_samples = images.size(0)  # 当前批的样本数量
    images = images.view(batch_samples, images.size(1), -1)  # 将图片展平
    total_images += batch_samples
    mean += images.mean(2).sum(0)  # 计算均值
    std += images.std(2).sum(0)    # 计算标准差

mean /= total_images
std /= total_images

print(f'Mean: {mean}, Std: {std}')