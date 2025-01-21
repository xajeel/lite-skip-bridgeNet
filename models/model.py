import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedModel(nn.Module):
    def __init__(self):
        super(OptimizedModel, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.extra_a = nn.Conv2d(3, 32, kernel_size=1, stride=1)
        self.extra_b = nn.Conv2d(32, 64, kernel_size=1, stride=1)

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        a = x
        x = self.conv1(x)
        b = x

        a = self.extra_a(a)
        x = b + a
        x = self.conv2(x)
        d = x

        a = self.extra_b(a)
        b = self.extra_b(b)
        d = F.interpolate(d, size=(128, 128), mode='bilinear', align_corners=False)

        x = a + b + d
        x = self.conv3(x)
        x = self.fc(x)
        return x
