#Cnn Setup

import torch
import torch.nn as nn
import torch.nn.functional as F

class RadarCNN(nn.Module):
    def __init__ (self, in_channels=1, num_classes=4, base=32, use_bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base, kernel_size=3,padding=1, bias=use_bias)
        self.bn1 = nn.BatchNorm2d(base)
        self.conv2 = nn.Conv2d(base, base*2,kernel_size=3,padding=1,bias=use_bias)
        self.bn2 = nn.BatchNorm2d(base*2)
        self.conv3 = nn.Conv2d(base*2,base*4,kernel_size=3,padding=1,bias=use_bias)
        self.bn3 = nn.BatchNorm2d(base*4)
        self.avg = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(base*4, num_classes, bias=True)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.avg_pool2d(x,2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x,2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.avg(x,2).flatten(1)

        return self.fc(x)

