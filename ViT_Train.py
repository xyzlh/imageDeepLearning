import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import Block


class ViTSeg(nn.Module):
    """Vision Transformer分割模型"""

    def __init__(self, in_channels=1, num_classes=1, img_size=224, patch_size=16, embed_dim=768, depth=6):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # 输入投影层（适配单通道输入）
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Transformer编码器
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=12) for _ in range(depth)
        ])

        # 解码器（上采样到原图尺寸）
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=4),  # 4倍上采样
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4),  # 再4倍上采样
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # 分块嵌入 [B, C, H, W] -> [B, D, H/P, W/P]
        x = self.proj(x)
        B, D, H, W = x.shape

        # 转换为序列 [B, N, D]
        x = rearrange(x, 'b d h w -> b (h w) d')

        # Transformer编码
        x = self.blocks(x)

        # 恢复空间维度 [B, N, D] -> [B, D, H/P, W/P]
        x = rearrange(x, 'b (h w) d -> b d h w', h=H, w=W)

        # 上采样解码
        x = self.decoder(x)
        return x
config = Config(
    device="cuda",
    root_dir="../archive/",
    train_img_dir="../archive/train_img",
    train_mask_dir="../archive/train_mask",
    test_img_dir="../archive/test_img",
    test_mask_dir="../archive/test_mask",
    valid_img_dir="../archive/valid_img",
    valid_mask_dir="../archive/valid_mask",
    backbone=ViTSeg(in_channels=1, img_size=224),  # 自定义ViT模型
    transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 单通道归一化
    ]),
    batchsize=4,
    lr=0.0001,  # ViT需要更小学习率
    num_epochs=10,
    print_freq=1
)


class TumorDataset(Dataset):
    def __getitem__(self, idx):
        img = Image.open(img_path).convert('L')  # 直接读取灰度图
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            img = self.transform(img)  # 单通道输入
            mask = self.transform(mask)  # 单通道标签

        return img, mask


def train(train_loader, valid_loader, model, criterion, optimizer, num_epochs):
    model.to(config.device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_iou = 0.0

        for i, (inputs, masks) in enumerate(tqdm(train_loader)):
            inputs = inputs.to(config.device)
            masks = masks.to(config.device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, masks)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算指标
            preds = (torch.sigmoid(outputs) > 0.5).float()
            iou = calculate_iou(preds.squeeze(1), masks.squeeze(1))
            running_loss += loss.item()
            running_iou += iou

        # 打印训练结果
        epoch_loss = running_loss / len(train_loader)
        epoch_iou = running_iou / len(train_loader)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, IoU: {epoch_iou:.4f}")


