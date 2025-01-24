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
import torch.nn.functional as F
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

class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DownsamplingBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, n_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        c = F.relu(self.conv1(x))
        c = F.relu(self.conv2(c))
        p = self.pool(c)
        return c, p


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(UpsamplingBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, n_filters, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(n_filters * 2, n_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)

    def forward(self, x, skip_connection):
        u = self.upconv(x)
        u = torch.cat([u, skip_connection], dim=1)  # Concatenate along channel dimension
        c = F.relu(self.conv1(u))
        c = F.relu(self.conv2(c))
        return c


class MultiheadAttentionBlock(nn.Module):
    def __init__(self, num_heads, key_dim):
        super(MultiheadAttentionBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=key_dim, num_heads=num_heads)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x_flattened = x.view(batch_size, channels, -1).permute(2, 0, 1)  # (H*W, N, C)
        attn_output, _ = self.multihead_attn(x_flattened, x_flattened, x_flattened)
        return attn_output.permute(1, 2, 0).view(batch_size, channels, height, width)


class UNetWithMultiHeadAttention(nn.Module):
    def __init__(self, input_channels=3, n_classes=1, n_filters=32, num_heads=4, key_dim=64):
        super(UNetWithMultiHeadAttention, self).__init__()

        self.dblock1 = DownsamplingBlock(input_channels, n_filters)
        self.dblock2 = DownsamplingBlock(n_filters, n_filters * 2)
        self.dblock3 = DownsamplingBlock(n_filters * 2, n_filters * 4)
        self.dblock4 = DownsamplingBlock(n_filters * 4, n_filters * 8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(n_filters * 8, n_filters * 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_filters * 16, n_filters * 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.attention = MultiheadAttentionBlock(num_heads, key_dim)

        self.u6 = UpsamplingBlock(n_filters * 16, n_filters * 8)
        self.u7 = UpsamplingBlock(n_filters * 8, n_filters * 4)
        self.u8 = UpsamplingBlock(n_filters * 4, n_filters * 2)


        self.u9 = UpsamplingBlock(n_filters * 2, n_filters)

        self.final_conv = nn.Conv2d(n_filters, n_classes, kernel_size=1)

    def forward(self, x):
        d1, p1 = self.dblock1(x)
        d2, p2 = self.dblock2(p1)
        d3, p3 = self.dblock3(p2)
        d4, p4 = self.dblock4(p3)

        # Pass through bottleneck
        bottleneck_features = self.bottleneck(p4)

        # Apply multi-head attention on the bottleneck features
        attention_output = self.attention(bottleneck_features)

        # Process upsampling blocks
        u6 = self.u6(attention_output, d4)
        u7 = self.u7(u6, d3)
        u8 = self.u8(u7, d2)
        u9 = self.u9(u8, d1)

        # Final output layer
        outputs = self.final_conv(u9)

        return outputs


# Define loss function and optimizer outside the model
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)

def combined_loss(y_true, y_pred):
    bce = nn.BCEWithLogitsLoss()(y_pred, y_true)
    dice = 1 - dice_coef(y_true, torch.sigmoid(y_pred))
    return 0.6 * dice + 0.4 * bce


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the model
model = UNetWithMultiHeadAttention(input_channels=3, n_classes=1, n_filters=32, key_dim=512).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
num_epochs = 5
plt.ion()  # Enable interactive mode for plotting

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    epoch_loss = 0
    correct_predictions = 0
    total_pixels = 0

    # Training phase
    for batch_index, (images, masks) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = combined_loss(masks, outputs)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # Calculate accuracy
        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()
        correct_predictions += (preds == masks).sum().item()
        total_pixels += torch.numel(preds)

    # Average loss and accuracy for train set
    avg_train_loss = epoch_loss / len(train_loader)
    train_accuracy = correct_predictions / total_pixels
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    val_correct_predictions = 0
    val_total_pixels = 0

    with torch.no_grad():  # Disable gradient calculation for validation
        for images, masks in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = combined_loss(masks, outputs)
            val_loss += loss.item()

            # Calculate accuracy
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()
            val_correct_predictions += (preds == masks).sum().item()
            val_total_pixels += torch.numel(preds)

    # Average loss and accuracy for validation set
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = val_correct_predictions / val_total_pixels
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Accuracy: {train_accuracy:.4f}, Train Loss: {avg_train_loss:.4f}, '
          f'Val Accuracy: {val_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, '
          f'Learning Rate: {current_lr:.6f}')

    # Real-time plotting of loss and accuracy
    plt.figure(figsize=(12, 5))

    # Plotting Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch + 2), train_losses, label='Train Loss', color='blue', marker='o')
    plt.plot(range(1, epoch + 2), val_losses, label='Validation Loss', color='orange', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()

    # Plotting Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch + 2), train_accuracies, label='Train Accuracy', color='blue', marker='o')
    plt.plot(range(1, epoch + 2), val_accuracies, label = 'Validation Accuracy', color='orange', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.pause(0.001)  # Pause to update the figure

# Save the model after training
model_path = './model/unet_model.pth'
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')

plt.ioff()  # Turn off interactive mode
plt.show()  # Display the final plots
