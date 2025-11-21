import torch
import torch.nn as nn


class ControllingParams(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 24, 3, padding=1)
        self.conv2 = nn.Conv2d(24, 48, 3, padding=1)
        self.reduce = nn.Conv2d(48, 24, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(24 * 4 * 4, num_classes)

    def forward(self, x): # <-[3, h, w]
        x = torch.relu(self.conv1(x)) # [24, h, w]
        x = torch.max_pool2d(x, 2) # [24, h/2, w/2]

        x = torch.relu(self.conv2(x)) # [48, h/2, w/2]
        x = torch.max_pool2d(x, 2) # [48, h/4, w/4]

        x = torch.relu(self.reduce(x)) # [24, h/4, w/4]
        x = torch.max_pool2d(x, 2) # [24, h/8, w/8]


        x = self.global_pool(x) # [24, h/16, w/16]

        x = x.view(x.size(0), -1) # 24 * 4 * 4 = 384
        return self.fc(x) # num_classes

model = ControllingParams(3, 10)
T_P = sum(p.numel() for p in model.parameters())
print(T_P)


