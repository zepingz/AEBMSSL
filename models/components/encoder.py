from torch import nn
from torchvision.models import resnet18

from .layers import ConvBlock


class DummyFrameEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(10, 5),)

    def forward(self, x):
        return self.model(x)


class TestFrameEncoder(nn.Module):
    _embedding_size = 64

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(1, 64, 7, 2, 3),
            ConvBlock(64, 64, 5, 2, 2),
            ConvBlock(64, 64, 5, 2, 2),
            ConvBlock(64, 64, 5, 2, 2),
            ConvBlock(64, 64, 5, 2, 2),
            ConvBlock(64, 64, 5, 2, 2),
        )

    def forward(self, x):
        bs = x.shape[0]
        x = self.layers(x)
        x = x.view(bs, -1)
        return x


class TestFrameEncoder2(nn.Module):
    _embedding_size = 64

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(1, 64, 7, 2, 3),
            ConvBlock(64, 64, 5, 2, 2),
            ConvBlock(64, 64, 5, 2, 2),
            ConvBlock(64, 64, 5, 2, 2),
            nn.Conv2d(64, 64, 5, 2, 2),
            nn.BatchNorm2d(64)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class ResNetFrameEncoder(nn.Module):
    _embedding_size = 512

    def __init__(self):
        super().__init__()
        self.resnet = nn.Sequential(*list(resnet18().children())[:-1])

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        return x
