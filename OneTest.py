import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from pycocotools.coco import COCO

from tqdm import tqdm
import matplotlib.pyplot as plt

class CocoDataset(Dataset):
    def __init__(self, annotation_file, image_dir, transform=None):
        self.coco = COCO(annotation_file)
        self.image_dir = image_dir
        self.image_ids = self.coco.getImgIds()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)

        for ann in anns:
            mask = np.maximum(mask, self.coco.annToMask(ann))

        if self.transform:

            image = self.transform(image)
            mask = self.transform(mask)  # 如果 mask 经过 transform，可以再定义合适的变换
        return image, mask
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset = CocoDataset('./train/_annotations.coco.json', './train', transform=transform)
val_dataset = CocoDataset('./valid/_annotations.coco.json', './valid', transform=transform)
test_dataset = CocoDataset('./test/_annotations.coco.json', './test', transform=transform)

BATCH_SIZE = 16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
class UNet(nn.Module):
    def __init__(self, n_classes=1):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Add more layers for the full U-Net architecture...
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x = self.decoder(x1)
        return x
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_classes=1).to(device)
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)

def combined_loss(y_true, y_pred):
    bce = nn.BCEWithLogitsLoss()(y_pred, y_true)
    dice = 1 - dice_coef(y_true, torch.sigmoid(y_pred))
    return 0.6 * dice + 0.4 * bce

optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 20



# 初始化用于存储每个 epoch 的损失值
losses = []

plt.ion()  # 开启交互模式

for epoch in range(num_epochs):
    model.train()  # 将模型设置为训练模式
    epoch_loss = 0  # 当前 epoch 的总损失

    # 使用 tqdm 包装 train_loader，以显示进度条
    for batch_index, (images, masks) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')):
        images = images.to(device)
        masks = masks.to(device)  # 确保 masks 在同一设备上
        optimizer.zero_grad()  # 清空梯度
        outputs = model(images)  # 前向传播
        loss = combined_loss(masks, outputs)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        epoch_loss += loss.item()  # 累加当前 batch 的损失

    avg_loss = epoch_loss / len(train_loader)  # 计算平均损失
    losses.append(avg_loss)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # 实时绘制损失曲线
    plt.figure(figsize=(6, 6))
    plt.plot(range(1, epoch + 2), losses, label='Loss', color='#76B900', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.pause(0.001)  # 暂停以更新图形

# 保存模型
model_path = 'unet_model.pth'
torch.save(model.state_dict(), model_path)
print(f'模型已保存至 {model_path}')

plt.ioff()  # 关闭交互模式
plt.show()  # 显示最终图形