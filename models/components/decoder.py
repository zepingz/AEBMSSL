from torch import nn

from .layers import ConvBlock


class DummyFrameDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(5, 10),)

    def forward(self, x):
        return self.model(x)


class TestFrameDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(64, 64, 4, 2, 1, transpose=True),  # 2x2
            ConvBlock(64, 64, 4, 2, 1, transpose=True),  # 4x4
            ConvBlock(64, 64, 4, 2, 1, transpose=True),  # 8x8
            ConvBlock(64, 64, 4, 2, 1, transpose=True),  # 16x16
            ConvBlock(64, 64, 4, 2, 1, transpose=True),  # 32x32
            nn.ConvTranspose2d(64, 1, 4, 2, 1),  # 64x64
        )

    def forward(self, x):
        x = x.view(-1, 64, 1, 1)
        x = self.layers(x)
        return x


class TestFrameDecoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # ConvBlock(64, 64, 4, 2, 1, transpose=True),  # 2x2
            ConvBlock(64, 64, 4, 2, 1, transpose=True),  # 4x4
            ConvBlock(64, 64, 4, 2, 1, transpose=True),  # 8x8
            ConvBlock(64, 64, 4, 2, 1, transpose=True),  # 16x16
            ConvBlock(64, 64, 4, 2, 1, transpose=True),  # 32x32
            nn.ConvTranspose2d(64, 1, 4, 2, 1),  # 64x64
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class ResNetFrameDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(512, 256, 4, 2, 1, transpose=True),  # 2x2
            ConvBlock(256, 128, 4, 2, 1, transpose=True),  # 4x4
            ConvBlock(128, 64, 4, 2, 1, transpose=True),  # 8x8
            ConvBlock(64, 64, 4, 2, 1, transpose=True),  # 16x16
            ConvBlock(64, 64, 4, 2, 1, transpose=True),  # 32x32
            ConvBlock(64, 64, 4, 2, 1, transpose=True),  # 64x64
            ConvBlock(64, 64, 4, 2, 1, transpose=True),  # 128x128
            # ConvBlock(64, 64, 4, 2, 1, transpose=True),  # 256x256
            nn.ConvTranspose2d(64, 3, 4, 2, 1),  # 512x512
        )

    def forward(self, x):
        x = x.view(-1, 512, 1, 1)
        x = self.layers(x)
        return x


# class ResNetFrameDecoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # TODO: this number is hard coded
#         self.fc = nn.Linear(512, 8192)
#         self.layers = nn.Sequential(
#             nn.Conv2d(512, 256, kernel_size=(3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(256, 128, kernel_size=(3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(64, 3, kernel_size=(1, 1), stride=1),
#         )
#
#     def forward(self, x):
#         x = self.fc(x)
#         x = x.view(-1, 512, 4, 4)
#         x = self.layers(x)
#         return x
