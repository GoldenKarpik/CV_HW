import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from PIL import Image

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# классификация

class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, class_names=None, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        with open(os.path.join(root_dir, 'wnids.txt'), 'r') as f:
            all_class_names = [line.strip() for line in f]

        if class_names is None:
            class_names = all_class_names
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.samples = self._make_dataset()
        print(f'{split}: {len(self.samples)} изображений, {len(self.class_names)} классов')

    def _make_dataset(self):
        data = []
        if self.split == 'train':
            train_dir = os.path.join(self.root_dir, 'train')
            for cls_name in self.class_names:
                img_dir = os.path.join(train_dir, cls_name, 'images')
                if not os.path.exists(img_dir):
                    continue
                for img_name in os.listdir(img_dir):
                    img_path = os.path.join(img_dir, img_name)
                    label = self.class_to_idx[cls_name]
                    data.append((img_path, label))
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 128x128 -> 64x64
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 64x64 -> 32x32
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 16x16 -> 8x8
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def train_epoch_classification(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc='обучение')
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (output.argmax(1) == target).sum().item()
        total += target.size(0)
        pbar.set_postfix({'loss': f'{total_loss / total:.4f}', 'acc': f'{100. * correct / total:.2f}%'})
    return total_loss / len(loader), 100. * correct / total


def validate_epoch_classification(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc='валидация')
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            correct += (output.argmax(1) == target).sum().item()
            total += target.size(0)
            pbar.set_postfix({'loss': f'{total_loss / total:.4f}', 'acc': f'{100. * correct / total:.2f}%'})
    return total_loss / len(loader), 100. * correct / total


def train_model_classification(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=20):
    print(f'\nобучение на {num_epochs} эпох\n')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    model = model.to(device)
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch_classification(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_epoch_classification(model, val_loader, criterion, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f'  train loss: {train_loss:.4f}, train acc: {train_acc:.2f}%')
        print(f'  val loss: {val_loss:.4f}, val acc: {val_acc:.2f}%')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_classifier.pth')
            print(f'лучшая модель: {val_acc:.2f}%')
    return history


def plot_classification_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history['train_loss'], label='train loss', color='blue', linewidth=2)
    ax1.plot(history['val_loss'], label='val loss', color='red', linewidth=2)
    ax1.set_title('потери')
    ax1.set_xlabel('эпоха')
    ax1.set_ylabel('loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(history['train_acc'], label='train acc', color='blue', linewidth=2)
    ax2.plot(history['val_acc'], label='val acc', color='red', linewidth=2)
    ax2.set_title('точность')
    ax2.set_xlabel('эпоха')
    ax2.set_ylabel('accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# сегментация

class MoonSegmentationDataset(Dataset):
    def __init__(self, root_dir, image_ids=None, augmentation=None, preprocessing=None):
        self.root_dir = root_dir
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        images_dir = os.path.join(root_dir, 'images', 'render')
        if image_ids is None:
            all_images = os.listdir(images_dir)
            self.image_ids = [img.replace('.png', '') for img in all_images if img.endswith('.png')]
        else:
            self.image_ids = image_ids

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.root_dir, 'images', 'render', f'{image_id}.png')
        mask_id = image_id.replace('render', '') if 'render' in image_id else image_id
        mask_path = os.path.join(self.root_dir, 'images', 'ground', f'ground{mask_id}.png')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        return image, mask


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[16, 32, 64, 128]):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        prev_channels = in_channels
        for feature in features:
            self.encoder_blocks.append(DoubleConv(prev_channels, feature))
            prev_channels = feature
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        for feature in reversed(features):
            self.decoder_blocks.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder_blocks.append(DoubleConv(feature * 2, feature))
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder_blocks), 2):
            x = self.decoder_blocks[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([skip_connection, x], dim=1)
            x = self.decoder_blocks[idx + 1](x)
        return self.final_conv(x)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, predictions, targets):
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def dice_coefficient(predictions, targets, threshold=0.5, smooth=1e-6):
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    intersection = (predictions * targets).sum()
    dice = (2. * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)
    return dice.item()


def iou_score(predictions, targets, threshold=0.5, smooth=1e-6):
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def pixel_accuracy(predictions, targets, threshold=0.5):
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    correct = (predictions == targets).float().sum()
    total = targets.numel()
    return (correct / total).item()


def train_epoch_segmentation(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    pbar = tqdm(loader, desc='обучение')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            dice = dice_coefficient(outputs, masks)
            iou = iou_score(outputs, masks)
        running_loss += loss.item()
        running_dice += dice
        running_iou += iou
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}', 'iou': f'{iou:.4f}'})
    return running_loss / len(loader), running_dice / len(loader), running_iou / len(loader)


def validate_epoch_segmentation(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_acc = 0.0
    pbar = tqdm(loader, desc='валидация')
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.unsqueeze(1).to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            dice = dice_coefficient(outputs, masks)
            iou = iou_score(outputs, masks)
            acc = pixel_accuracy(outputs, masks)
            running_loss += loss.item()
            running_dice += dice
            running_iou += iou
            running_acc += acc
            pbar.set_postfix(
                {'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}', 'iou': f'{iou:.4f}', 'acc': f'{acc:.4f}'})
    return running_loss / len(loader), running_dice / len(loader), running_iou / len(loader), running_acc / len(loader)


def train_model_segmentation(model, train_loader, val_loader, optimizer, criterion, scheduler, device, num_epochs=20,
                             save_path='best_unet.pth'):
    model = model.to(device)
    history = {'train_loss': [], 'train_dice': [], 'train_iou': [], 'val_loss': [], 'val_dice': [], 'val_iou': [],
               'val_acc': []}
    best_val_dice = 0.0
    for epoch in range(num_epochs):
        print(f'эпоха {epoch + 1}/{num_epochs}')
        train_loss, train_dice, train_iou = train_epoch_segmentation(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice, val_iou, val_acc = validate_epoch_segmentation(model, val_loader, criterion, device)
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        history['val_acc'].append(val_acc)
        if scheduler:
            scheduler.step(val_dice)
        print(f'  train - loss: {train_loss:.4f}, dice: {train_dice:.4f}, iou: {train_iou:.4f}')
        print(f'  val   - loss: {val_loss:.4f}, dice: {val_dice:.4f}, iou: {val_iou:.4f}, acc: {val_acc:.4f}')
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_iou': val_iou,
            }, save_path)
            print(f'лучшая модель с dice: {val_dice:.4f}')
    print(f'лучший val dice: {best_val_dice:.4f}')
    return history


def plot_segmentation_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(history['train_loss'], label='train loss', color='blue', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='val loss', color='red', linewidth=2)
    axes[0, 0].set_title('loss')
    axes[0, 0].set_xlabel('эпоха')
    axes[0, 0].set_ylabel('loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(history['train_dice'], label='train dice', color='blue', linewidth=2)
    axes[0, 1].plot(history['val_dice'], label='val dice', color='red', linewidth=2)
    axes[0, 1].set_title('dice coefficient')
    axes[0, 1].set_xlabel('эпоха')
    axes[0, 1].set_ylabel('dice')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(history['train_iou'], label='train iou', color='blue', linewidth=2)
    axes[1, 0].plot(history['val_iou'], label='val iou', color='red', linewidth=2)
    axes[1, 0].set_title('iou score')
    axes[1, 0].set_xlabel('эпоха')
    axes[1, 0].set_ylabel('iou')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(history['val_acc'], label='val acc', color='red', linewidth=2)
    axes[1, 1].set_title('pixel accuracy')
    axes[1, 1].set_xlabel('эпоха')
    axes[1, 1].set_ylabel('accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor * std + mean
    return img.clamp(0, 1)


def visualize_segmentation_predictions(model, loader, device, num_samples=5):
    model.eval()
    images, masks = next(iter(loader))
    images = images.to(device)
    masks = masks.to(device)
    with torch.no_grad():
        outputs = model(images)
        predictions = torch.sigmoid(outputs)
        predictions = (predictions > 0.5).float()
    images = images.cpu()
    predictions = predictions.cpu()

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    for i in range(min(num_samples, len(images))):
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        mask_true = masks[i].cpu().numpy()
        mask_pred = predictions[i, 0].numpy()

        axes[i, 0].imshow(img)
        axes[i, 0].set_title('исходное изображение')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask_true, cmap='gray')
        axes[i, 1].set_title('истинная маска')
        axes[i, 1].axis('off')

        dice_val = dice_coefficient(outputs[i:i + 1], masks[i:i + 1].unsqueeze(1))
        iou_val = iou_score(outputs[i:i + 1], masks[i:i + 1].unsqueeze(1))
        axes[i, 2].imshow(mask_pred, cmap='gray')
        axes[i, 2].set_title(f'предсказание\ndice: {dice_val:.3f}, iou: {iou_val:.3f}')
        axes[i, 2].axis('off')

        overlay = img.copy()
        overlay[mask_pred > 0.5] = [0, 1, 0]
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('наложение')
        axes[i, 3].axis('off')
    plt.suptitle('предсказания модели u-net', fontsize=16)
    plt.tight_layout()
    plt.show()


#unet с бэкбоном

class UNetWithBackbone(nn.Module):
    def __init__(self, backbone, out_channels=1):
        super().__init__()
        self.backbone = backbone
        self.bottleneck = DoubleConv(512, 1024)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(512 + 512, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(256 + 256, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(128 + 128, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(64 + 64, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder (skip connections)
        enc1 = F.relu(self.backbone.bn1(self.backbone.conv1(x)))
        x = self.backbone.pool(enc1)

        enc2 = F.relu(self.backbone.bn2(self.backbone.conv2(x)))
        x = self.backbone.pool(enc2)

        enc3 = F.relu(self.backbone.bn3(self.backbone.conv3(x)))
        x = self.backbone.pool(enc3)

        enc4 = F.relu(self.backbone.bn4(self.backbone.conv4(x)))
        x = self.backbone.pool(enc4)

        bottleneck = self.bottleneck(x)

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = self.decoder4(torch.cat([dec4, enc4], dim=1))

        dec3 = self.upconv3(dec4)
        dec3 = self.decoder3(torch.cat([dec3, enc3], dim=1))

        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat([dec2, enc2], dim=1))

        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat([dec1, enc1], dim=1))

        return self.final_conv(dec1)



if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)

    # Классификация
    root_dir = r'D:\projects\ComputerVision_HW\tiny-imagenet-200'

    with open(os.path.join(root_dir, 'wnids.txt'), 'r') as f:
        all_class_names = [line.strip() for line in f]
    selected_classes = all_class_names[:20]
    print(f'выбранные классы: {selected_classes}')

    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = TinyImageNetDataset(root_dir, selected_classes, split='train', transform=train_transform)
    all_labels = [lbl for _, lbl in full_dataset.samples]
    train_idx, val_idx = train_test_split(range(len(full_dataset)), test_size=0.2, stratify=all_labels, random_state=42)
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(
        TinyImageNetDataset(root_dir, selected_classes, split='train', transform=val_transform), val_idx)
    train_loader_class = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_loader_class = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    print(f'тренировочных изображений: {len(train_dataset)}')
    print(f'валидационных изображений: {len(val_dataset)}')

    classifier = SimpleCNN(num_classes=len(selected_classes)).to(device)
    print(f'параметров: {count_parameters(classifier):,}')

    optimizer_class = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion_class = nn.CrossEntropyLoss()
    history_class = train_model_classification(classifier, train_loader_class, val_loader_class, optimizer_class,
                                               criterion_class, device, num_epochs=30)
    plot_classification_history(history_class)

    # Сегментация - базовая U-Net
    data_root = r'D:\projects\ComputerVision_HW\MOON_SEGMENTATION_BINARY'
    train_aug = A.Compose([
        A.Resize(128, 128),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=0.1, scale=0.1, rotate=45, p=0.5),
        A.OneOf([A.GaussNoise(var_limit=(10.0, 50.0)), A.GaussianBlur(blur_limit=(3, 7))], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ])
    val_aug = A.Compose([A.Resize(128, 128)])
    preprocessing = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    images_dir = os.path.join(data_root, 'images', 'render')
    all_images = [img.replace('.png', '') for img in os.listdir(images_dir) if img.endswith('.png')]
    train_ids, val_ids = train_test_split(all_images, test_size=0.2, random_state=42)
    train_dataset_moon = MoonSegmentationDataset(data_root, image_ids=train_ids, augmentation=train_aug,
                                                 preprocessing=preprocessing)
    val_dataset_moon = MoonSegmentationDataset(data_root, image_ids=val_ids, augmentation=val_aug,
                                               preprocessing=preprocessing)
    train_loader_moon = DataLoader(train_dataset_moon, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader_moon = DataLoader(val_dataset_moon, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    print(f'тренировочных изображений: {len(train_dataset_moon)}')
    print(f'валидационных изображений: {len(val_dataset_moon)}')

    unet = UNet(in_channels=3, out_channels=1, features=[16, 32, 64, 128]).to(device)
    print(f'параметров базовой u-net: {count_parameters(unet):,}')

    optimizer_unet = torch.optim.Adam(unet.parameters(), lr=1e-3)
    criterion_unet = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    scheduler_unet = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_unet, mode='max', factor=0.5, patience=3)
    history_unet = train_model_segmentation(unet, train_loader_moon, val_loader_moon, optimizer_unet, criterion_unet,
                                            scheduler_unet, device, num_epochs=30, save_path='models/best_unet.pth')
    plot_segmentation_history(history_unet)
    visualize_segmentation_predictions(unet, val_loader_moon, device)

    # U-Net с бэкбоном (замороженный)
    backbone_frozen = SimpleCNN(num_classes=len(selected_classes))
    backbone_frozen.load_state_dict(torch.load('models/best_classifier.pth', map_location='cpu', weights_only=False))
    unet_frozen = UNetWithBackbone(backbone_frozen, out_channels=1).to(device)

    for param in unet_frozen.backbone.parameters():
        param.requires_grad = False

    optimizer_frozen = torch.optim.Adam(filter(lambda p: p.requires_grad, unet_frozen.parameters()), lr=1e-3)
    history_frozen = train_model_segmentation(unet_frozen, train_loader_moon, val_loader_moon, optimizer_frozen,
                                              criterion_unet, None, device, num_epochs=30,
                                              save_path='models/best_unet_frozen.pth')
    plot_segmentation_history(history_frozen)

    # U-Net с бэкбоном (размороженный)
    backbone_unfrozen = SimpleCNN(num_classes=len(selected_classes))
    backbone_unfrozen.load_state_dict(torch.load('models/best_classifier.pth', map_location='cpu', weights_only=False))
    unet_unfrozen = UNetWithBackbone(backbone_unfrozen, out_channels=1).to(device)

    backbone_params = []
    decoder_params = []
    for name, param in unet_unfrozen.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            decoder_params.append(param)

    optimizer_unfrozen = torch.optim.Adam([
        {'params': backbone_params, 'lr': 1e-4},
        {'params': decoder_params, 'lr': 1e-3}
    ])
    scheduler_unfrozen = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_unfrozen, mode='max', factor=0.5,
                                                                    patience=3)
    history_unfrozen = train_model_segmentation(unet_unfrozen, train_loader_moon, val_loader_moon, optimizer_unfrozen,
                                                criterion_unet, scheduler_unfrozen, device, num_epochs=30,
                                                save_path='models/best_unet_unfrozen.pth')
    plot_segmentation_history(history_unfrozen)