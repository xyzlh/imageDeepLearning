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
optimizer = optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Initialize lists to store losses and accuracies for each epoch
losses = []
accuracies = []
num_epochs=5
plt.ion()  # Enable interactive mode for plotting

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    epoch_loss = 0  # Total loss for the current epoch
    correct_predictions = 0  # Counter for correct predictions
    total_pixels = 0  # Counter for total pixels processed

    # Use tqdm to show progress of train_loader
    for batch_index, (images, masks) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')):
        images = images.to(device)
        masks = masks.to(device)  # Ensure masks are on the same device
        optimizer.zero_grad()  # Clear gradients
        outputs = model(images)  # Forward pass
        loss = combined_loss(masks, outputs)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters
        epoch_loss += loss.item()  # Accumulate the loss for the epoch

        # Calculate accuracy (assuming binary classification with sigmoid activation)
        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()  # Convert probabilities to binary predictions
        correct_predictions += (preds == masks).sum().item()
        total_pixels += torch.numel(preds)
    scheduler.step()
    avg_loss = epoch_loss / len(train_loader)  # Average loss for the epoch
    accuracy = correct_predictions / total_pixels  # Calculate accuracy
    losses.append(avg_loss)
    accuracies.append(accuracy)
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch [{epoch + 1}/{num_epochs}], Accuracy: {accuracy:.4f},Loss: {avg_loss:.4f}, Learning Rate: {current_lr:.6f}')


    # Real-time plotting of loss and accuracy
    plt.figure(figsize=(6, 6))
    plt.plot(range(1, epoch + 2), losses, label='Loss', color='#76B900', marker='o')
    plt.plot(range(1, epoch + 2), accuracies, label='Accuracy', color='orange', marker='x')  # Accuracy plot
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Training Loss and Accuracy')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.pause(0.001)  # Pause to update the figure

# Save the model after training
model_path = './model/unet_model.pth'
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')

plt.ioff()  # Turn off interactive mode
plt.show()  # Display the final plot
