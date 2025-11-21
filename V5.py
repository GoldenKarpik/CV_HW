import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class FlexibleGradientNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()

        # 6 сверточных слоев с ReLU
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(8)
        self.fc = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x):

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))

        x = self.adaptive_pool(x)

        x = x.view(x.size(0), -1)

        return self.fc(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

input_size = 512


model = FlexibleGradientNet(input_channels=3, num_classes=10).to(device)
x = torch.randn(4, 3, input_size, input_size).to(device)
target = torch.randint(0, 10, (4,)).to(device)


output = model(x)
loss = nn.CrossEntropyLoss()(output, target)

#сохраняем градиенты
model.zero_grad()
loss.backward()

T_P = sum(p.numel() for p in model.parameters())
print( T_P)

# собираем нормы градиентов
layer_names = []
grad_norms = []

layers = [model.conv1, model.conv2, model.conv3, model.conv4, model.conv5, model.conv6]

for i, layer in enumerate(layers, 1):
    grad_norm = layer.weight.grad.norm().item()
    layer_names.append(f'conv{i}')
    grad_norms.append(grad_norm)
    print(f"conv{i}: {grad_norm:.8f}")

plt.figure(figsize=(10, 6))
plt.bar(layer_names, grad_norms)
plt.yscale('log')
plt.show()

# Вывод параметров
"""
Даже если бы значения градиентов бли бы примерно равны, то на пооследенм слое из за колличества параметров их было бы куда больше, в следствии чего при рассчете градиетов из за колличества слагаемых результат будет больш. 
Еще есьт версия о том что градиенты мы считаем в обратную сторону и когда мы будем считать градиенты 1 слоя из за большого колличества множителей (которые могут быть близкими к 0 из за ReLU) мы сильно приближаем значение к 0, хотя скорей всего это имеет не особо большой эффект так как при иницилизации весов стараются сохранить дисперсию из за чего средний вес 1
"""



