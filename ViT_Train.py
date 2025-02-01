import os
import torch
import timm
import torch.nn as nn
from timm import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torch.amp import GradScaler, autocast  # 使用新版AMP API


# ==================== ViT分割模型 ====================
class ViTSeg(nn.Module):
    def __init__(self, num_classes=1, img_size=224):
        super().__init__()

        # 加载预训练ViT（临时使用3通道加载权重）
        self.vit = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=True,
            in_chans=1,
            img_size=img_size
        )

        # 适配单通道输入
        self._adapt_first_conv()

        # 移除class token相关参数
        del self.vit.cls_token
        self.vit.pos_embed = nn.Parameter(self.vit.pos_embed[:, 1:, :])  # 移除分类位置编码

        # 冻结前6层Transformer
        self._freeze_layers(6)

        # 自定义解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(192, 128, 4, 4),  # 输入通道需与ViT输出维度一致
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ConvTranspose2d(64, num_classes, 4, 4)
        )

    def _adapt_first_conv(self):
        """修改首层卷积适配单通道输入"""
        orig_conv = self.vit.patch_embed.proj
        new_conv = nn.Conv2d(
            1,  # 修改为单通道输入
            orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding
        )

        # 平均三通道权重到单通道
        with torch.no_grad():
            new_conv.weight.data = orig_conv.weight.data.mean(dim=1, keepdim=True)
            new_conv.bias.data = orig_conv.bias.data.clone()

        self.vit.patch_embed.proj = new_conv

    def _freeze_layers(self, num_layers):
        """冻结前N层Transformer"""
        for i, block in enumerate(self.vit.blocks):
            if i < num_layers:
                for param in block.parameters():
                    param.requires_grad = False

    def forward(self, x):
        # 1. Patch Embedding
        x = self.vit.patch_embed(x)  # [B, N, C]

        # 2. 添加位置编码
        x = x + self.vit.pos_embed

        # 3. Transformer处理
        x = self.vit.blocks(x)

        # 4. 转换为空间特征图
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.permute(0, 2, 1).view(B, C, H, W)

        # 5. 解码器上采样
        return self.decoder(x)


# ==================== 数据管道 ====================
class TumorDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_files = sorted(os.listdir(img_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        img = Image.open(img_path).convert('L')  # 单通道输入
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            img = self.transform(img)
            mask = (self.transform(mask) > 0.5).float()  # 直接二值化

        return img, mask


# ==================== 训练引擎 ====================
class ViTTrainer:
    def __init__(self, config):
        self.config = config
        self.scaler = GradScaler()  # 新版GradScaler
        self.best_iou = 0.0

        # 初始化模型
        self.model = ViTSeg(img_size=224).to(config.device)

        # 数据加载
        self._init_dataloaders()

        # 优化配置
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.lr,
            weight_decay=1e-4
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def _init_dataloaders(self):
        """数据加载器（显存优化配置）"""
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        val_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.train_loader = DataLoader(
            TumorDataset(self.config.train_img_dir, self.config.train_mask_dir, train_transform),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            TumorDataset(self.config.valid_img_dir, self.config.valid_mask_dir, val_transform),
            batch_size=self.config.batch_size,
            shuffle=False
        )

    def _calculate_iou(self, preds, targets):
        # 确保输入为二值张量（0或1）
        preds_bool = preds.bool()  # 若 preds 是浮点型（0.0/1.0），直接转布尔型
        targets_bool = targets.bool()

        intersection = (preds_bool & targets_bool).float().sum((1, 2))  # 按批次计算交集
        union = (preds_bool | targets_bool).float().sum((1, 2))  # 按批次计算并集
        iou = (intersection + 1e-6) / (union + 1e-6)  # 防止除以零
        return iou.mean().item()  # 返回平均IoU

    def train_epoch(self, epoch):
        """训练单个epoch"""
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
        for inputs, masks in progress_bar:
            inputs = inputs.to(self.config.device)
            masks = masks.to(self.config.device)

            self.optimizer.zero_grad()

            # 混合精度训练
            with autocast(device_type='cuda'):  # 新版autocast
                outputs = self.model(inputs)
                loss = self.criterion(outputs, masks)

            # 反向传播
            self.scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 统计指标
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            iou = self._calculate_iou(preds, masks)
            progress_bar.set_postfix(loss=loss.item(), iou=iou)

        return total_loss / len(self.train_loader)

    def validate(self):
        """验证过程"""
        self.model.eval()
        total_iou = 0.0

        with torch.no_grad():
            for inputs, masks in self.val_loader:
                inputs = inputs.to(self.config.device)
                masks = masks.to(self.config.device)

                outputs = self.model(inputs)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                total_iou += self._calculate_iou(preds, masks)

        return total_iou / len(self.val_loader)

    def train(self):
        """主训练循环"""
        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_iou = self.validate()

            print(f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                  f"Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f}")

            # 保存最佳模型
            if val_iou > self.best_iou:
                self.best_iou = val_iou
                torch.save(self.model.state_dict(), f'./model/ViT_epoch_{config.num_epochs}.pth')



# ==================== 配置文件 ====================
class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_img_dir = "../archive/train_img"
        self.train_mask_dir = "../archive/train_mask"
        self.valid_img_dir = "../archive/valid_img"
        self.valid_mask_dir = "../archive/valid_mask"
        self.batch_size = 2  # 4GB显存建议设为2
        self.lr = 1e-4  # ViT微调推荐学习率
        self.num_epochs = 3
        self.transform=transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])
        self.backbone=ViTSeg(img_size=224).to(self.device)
config = Config()
# ==================== 执行训练 ====================
if __name__ == "__main__":

    trainer = ViTTrainer(config)
    trainer.train()