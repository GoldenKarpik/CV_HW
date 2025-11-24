import os
import random
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, class_names=None, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # загружаем все классы
        with open(os.path.join(root_dir, 'wnids.txt'), 'r') as f:
            all_class_names = [line.strip() for line in f]

        # выбираем подмножество классов
        if class_names is None:
            class_names = all_class_names
        self.class_names = class_names

        # создаем маппинг имен в индексы
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        # загружаем образцы
        self.samples = self._make_dataset()

        print(f"Создан датасет {split}: {len(self.samples)} образцов, {len(self.class_names)} классов")

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

        elif self.split == 'val':
            val_dir = os.path.join(self.root_dir, 'val')
            img_dir = os.path.join(val_dir, 'images')
            anno_path = os.path.join(val_dir, 'val_annotations.txt')

            # создаем маппинг изображение-класс
            label_map = {}
            with open(anno_path, 'r') as f:
                for line in f:
                    img_name, cls_name, *_ = line.strip().split('\t')
                    if cls_name in self.class_names:
                        label_map[img_name] = self.class_to_idx[cls_name]

            for img_name in os.listdir(img_dir):
                if img_name in label_map:
                    img_path = os.path.join(img_dir, img_name)
                    data.append((img_path, label_map[img_name]))

        elif self.split == 'test':
            test_dir = os.path.join(self.root_dir, 'test', 'images')
            for img_name in os.listdir(test_dir):
                img_path = os.path.join(test_dir, img_name)
                data.append((img_path, -1))  # тест без меток

        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def denormalize(img_tensor):
    # изображение из нормализовнного состояния
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
    img = img_tensor.permute(1, 2, 0) * std + mean
    return img.clamp(0, 1)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, activation='relu'):
        super().__init__()

        # сохраняем тип активации для использования в forward
        self.activation_type = activation
        self.activation = self._get_activation(activation)

        # первый сверточный слой
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # второй сверточный слой
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # skip connection если вдруг размерносить не совпадут
        self.downsample = downsample

    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.01, inplace=True)
        elif activation == 'elu':
            return nn.ELU(alpha=1.0, inplace=True)
        elif activation == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU(inplace=True)

    def forward(self, x):
        # сохраняем вход для skip connection
        identity = x

        # первый сверток + батч-норм + активация
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        # второй сверток + батч-норм
        out = self.conv2(out)
        out = self.bn2(out)

        # skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        # складываем и применяем активацию
        out += identity
        out = self.activation(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, channels=None, num_blocks=None, activation='relu'):
        super().__init__()

        if channels is None:
            channels = [64, 128, 256, 512]
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]

        # проверка длины
        assert len(channels) == len(num_blocks)

        self.channels = channels
        self.num_blocks = num_blocks

        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.activation = self._get_activation(activation)

        # residual слои
        self.layers = nn.ModuleList()
        in_channels = channels[0]

        for i, (out_channels, num_block) in enumerate(zip(channels, num_blocks)):
            # для первого слоя stride=1, для остальных stride=2
            stride = 1 if i == 0 else 2
            layer = self._make_layer(in_channels, out_channels, num_block, stride, activation)
            self.layers.append(layer)
            in_channels = out_channels

        # адаптивный pooling и классификатор
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], num_classes)
        self._initialize_weights()

    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.01, inplace=True)
        elif activation == 'elu':
            return nn.ELU(alpha=1.0, inplace=True)
        elif activation == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU(inplace=True)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, activation):
        layers = []

        # первый блок может иметь downsample и stride=2
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers.append(BasicBlock(in_channels, out_channels, stride, downsample, activation))

        # остальные блоки (stride=1, без downsample)
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, activation=activation))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        # x = self.maxpool(x)

        # residual блоки
        for layer in self.layers:
            x = layer(x)

        # классификация
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Обучение")
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

        pbar.set_postfix({
            'loss': f'{total_loss / total:.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    return total_loss / len(loader), 100. * correct / total


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Валидация")
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            correct += (output.argmax(1) == target).sum().item()
            total += target.size(0)

            pbar.set_postfix({
                'loss': f'{total_loss / total:.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

    return total_loss / len(loader), 100. * correct / total


def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=20):
    print(f"\n {num_epochs} эпох\n")

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    best_val_acc = 0.0

    model = model.to(device)

    for epoch in range(num_epochs):
        print(f"Эпоха {epoch + 1}/{num_epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  лучшая модель: {val_acc:.2f}%\n")

    return history


def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # график потерь
    ax1.plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', color='red', linewidth=2)
    ax1.set_title('Потери')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # график точности
    ax2.plot(history['train_acc'], label='Train Acc', color='blue', linewidth=2)
    ax2.plot(history['val_acc'], label='Val Acc', color='red', linewidth=2)
    ax2.set_title('Точность')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_experiment(model, num_epochs=20, model_name='model'):
    # считаем параметры
    params = count_parameters(model)
    print(f"Параметры: {params:,}\n")

    # вывод модели
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    history = train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs)

    # сохраняем модель
    torch.save(model.state_dict(), f'{model_name}.pth')
    print(f"модель сохранена: {model_name}.pth\n")

    # график
    plot_training_history(history)

    return {
        'params': params,
        'train_acc': history['train_acc'][-1],
        'val_acc': history['val_acc'][-1],
        'history': history
    }


def visualize_predictions(model, loader, num_samples=10, title=""):
    model.eval()
    plt.figure(figsize=(15, 8))
    images_shown = 0

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            outputs = model(data)
            _, predictions = torch.max(outputs, 1)

            for i in range(data.size(0)):
                if images_shown >= num_samples or targets[i] == -1:
                    continue

                plt.subplot(2, 5, images_shown + 1)
                img_vis = denormalize(data[i].cpu())
                plt.imshow(img_vis)

                true_label = targets[i].item()
                pred_label = predictions[i].item()
                color = 'green' if true_label == pred_label else 'red'

                plt.title(f"Истинный: {true_label}\nПрдскзнный: {pred_label}", color=color)
                plt.axis('off')
                images_shown += 1

            if images_shown >= num_samples:
                break

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    root_dir = r"D:\projects\ComputerVision_HW\tiny-imagenet-200"

    with open(os.path.join(root_dir, 'wnids.txt'), 'r') as f:
        all_class_names = [line.strip() for line in f]

    # выбираем 10 классов (можно выбрать любые, но мне лень, берем первые 10)
    selected_classes = all_class_names[:10]
    print(f"Выбранные классы:{selected_classes}")

    # проверяем, сколько изображений у нас есть для этих классов
    train_samples = []
    for cls in selected_classes:
        img_dir = os.path.join(root_dir, 'train', cls, 'images')
        if os.path.exists(img_dir):
            count = len(os.listdir(img_dir))
            train_samples.append((cls, count))
            print(f"Класс {cls}: {count} изображений")

    # Трансформации для обучения и валидации
    train_transform = transforms.Compose([
        transforms.Resize((72, 72)),
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Создаем датасеты
    train_dataset = TinyImageNetDataset(root_dir, selected_classes, split='train', transform=train_transform)
    val_dataset = TinyImageNetDataset(root_dir, selected_classes, split='val', transform=val_transform)
    test_dataset = TinyImageNetDataset(root_dir, selected_classes, split='test', transform=val_transform)

    print(f"\ntrain: {len(train_dataset)}")
    print(f"val: {len(val_dataset)}")
    print(f"test: {len(test_dataset)}")

    # Создаем DataLoader'ы
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    # Показываем несколько примеров из обучающего набора
    images, labels = next(iter(train_loader))

    plt.figure(figsize=(15, 6))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        img_vis = denormalize(images[i])
        plt.imshow(img_vis)
        plt.title(f"Class: {labels[i].item()}")
        plt.axis('off')
    plt.suptitle("Примеры изображений из выбранных 10 классов", fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.pause(0.001)

    print(f"размер батча: {images.shape}")
    print(f"диапазон значений: [{images.min():.3f}, {images.max():.3f}]")

    # 3.1
    baseline_model = ResNet18(
        num_classes=10,
        channels=[64, 128, 256],
        num_blocks=[2, 2, 2],
        activation='relu'
    ).to(device)

    baseline_result = run_experiment(baseline_model, num_epochs=20, model_name='baseline_model')

    # 3.2
    model_blocks_A = ResNet18(
        num_classes=10,
        channels=[32, 64, 128, 256],
        num_blocks=[1, 1, 1, 1],
        activation='relu'
    ).to(device)
    # [1,1,1,1] блоков
    result_blocks_A = run_experiment(model_blocks_A, num_epochs=20, model_name='model_blocks_A')

    # 3.2-B (добавлено)
    model_blocks_B = ResNet18(
        num_classes=10,
        channels=[32, 64, 128, 256],
        num_blocks=[2, 2, 2, 2],
        activation='relu'
    ).to(device)
    result_blocks_B = run_experiment(model_blocks_B, num_epochs=20, model_name='model_blocks_B')

    # 3.2-C (добавлено)
    model_blocks_C = ResNet18(
        num_classes=10,
        channels=[32, 64, 128, 256],
        num_blocks=[3, 3, 3, 3],
        activation='relu'
    ).to(device)
    result_blocks_C = run_experiment(model_blocks_C, num_epochs=20, model_name='model_blocks_C')

    # 4
    activations = {
        '3.3-A': ('ReLU', 'relu'),
        '3.3-B': ('LeakyReLU', 'leaky_relu'),
        '3.3-C': ('ELU', 'elu'),
        '3.3-D': ('GELU', 'gelu'),
    }

    activation_results = {}

    for exp_key, (name, act_func) in activations.items():
        model_act = ResNet18(
            num_classes=10,
            channels=[32, 64, 128, 256],
            num_blocks=[2, 2, 2, 2],
            activation=act_func
        ).to(device)

        result_act = run_experiment(model_act, num_epochs=20, model_name=f'model_{exp_key}')
        activation_results[exp_key] = result_act

    # 5
    final_model = ResNet18(
        num_classes=10,
        channels=[32, 64, 128, 256],
        num_blocks=[2, 2, 2, 2],
        activation='leaky_relu'  # показала лучшие результаты
    ).to(device)

    final_result = run_experiment(final_model, num_epochs=35, model_name='final_model')  # 30-40 эпох

    # тестирование на test set
    final_model.load_state_dict(torch.load('best_model.pth'))
    final_model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        with tqdm(test_loader) as pbar:
            for data, target in pbar:
                data = data.to(device)
                output = final_model(data)
                _, predicted = torch.max(output, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(target.numpy())

    # убираем пустые метки (-1)
    valid_indices = [i for i, label in enumerate(all_labels) if label != -1]
    if valid_indices:
        test_preds = [all_preds[i] for i in valid_indices]
        test_labels = [all_labels[i] for i in valid_indices]

        print(f"\nTest Accuracy: {100 * sum(p == l for p, l in zip(test_preds, test_labels)) / len(test_labels):.2f}%")
        print(classification_report(test_labels, test_preds, target_names=[f"Class_{i}" for i in range(10)]))

        # confusion matrix (добавлено)
        cm = confusion_matrix(test_labels, test_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f"Class_{i}" for i in range(10)],
                    yticklabels=[f"Class_{i}" for i in range(10)])
        plt.title('Confusion Matrix')
        plt.ylabel('истинный класс')
        plt.xlabel('предсказанный класс')
        plt.show()

    # визуализация с исправленными подписями
    visualize_predictions(final_model, test_loader, title="Задание 5: Финальная модель LeakyReLU")

    # собираем все результаты
    all_results_data = [
        {
            'Этап': 'Baseline',
            'Конфигурация': '[64,128,256]',
            'Параметры': f"{baseline_result['params']:,}",
            'Val Accuracy': f"{baseline_result['val_acc']:.2f}%",
            'Train Accuracy': f"{baseline_result['train_acc']:.2f}%"
        },
        {
            'Этап': '3.2-A',
            'Конфигурация': '[1,1,1,1] блоков',
            'Параметры': f"{result_blocks_A['params']:,}",
            'Val Accuracy': f"{result_blocks_A['val_acc']:.2f}%",
            'Train Accuracy': f"{result_blocks_A['train_acc']:.2f}%"
        },
        {
            'Этап': '3.2-B',
            'Конфигурация': '[2,2,2,2] блоков',
            'Параметры': f"{result_blocks_B['params']:,}",
            'Val Accuracy': f"{result_blocks_B['val_acc']:.2f}%",
            'Train Accuracy': f"{result_blocks_B['train_acc']:.2f}%"
        },
        {
            'Этап': '3.2-C',
            'Конфигурация': '[3,3,3,3] блоков',
            'Параметры': f"{result_blocks_C['params']:,}",
            'Val Accuracy': f"{result_blocks_C['val_acc']:.2f}%",
            'Train Accuracy': f"{result_blocks_C['train_acc']:.2f}%"
        },
    ]

    for exp_key, result in activation_results.items():
        name = activations[exp_key][0]
        all_results_data.append({
            'Этап': exp_key,
            'Конфигурация': name,
            'Параметры': f"{result['params']:,}",
            'Val Accuracy': f"{result['val_acc']:.2f}%",
            'Train Accuracy': f"{result['train_acc']:.2f}%"
        })

    all_results_data.append({
        'Этап': 'Final',
        'Конфигурация': '32 64 128 256 + LeakyReLU',
        'Параметры': f"{final_result['params']:,}",
        'Val Accuracy': f"{final_result['val_acc']:.2f}%",
        'Train Accuracy': f"{final_result['train_acc']:.2f}%"
    })

    df_results = pd.DataFrame(all_results_data)
    df_results = df_results.sort_values('Val Accuracy', ascending=False)

    print("\nИтоговая таблица всех экспериментов:")
    print(df_results.to_string(index=False))

    df_results.to_csv('experiment_results.csv', index=False)