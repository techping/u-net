# File: model.py
# U-Net Model Implementation
# Ziping Chen

import torch
import torch.nn as nn

def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
    )

def upconv(in_c, out_c):
    return nn.ConvTranspose2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=2,
            stride=2
        )

def crop(in_tensor, out_tensor):
    diff = (in_tensor.size()[2] - out_tensor.size()[2]) // 2
    return in_tensor[:, :, diff:-diff, diff:-diff]

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.contracting_path()
        self.expansive_path()

    def contracting_path(self):
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(1, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

    def expansive_path(self):
        self.up_conv_1 = upconv(1024, 512)
        self.up_conv_2 = upconv(512, 256)
        self.up_conv_3 = upconv(256, 128)
        self.up_conv_4 = upconv(128, 64)
        self.down_conv_6 = double_conv(1024, 512)
        self.down_conv_7 = double_conv(512, 256)
        self.down_conv_8 = double_conv(256, 128)
        self.down_conv_9 = double_conv(128, 64)
        self.conv_1x1 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)

    def forward(self, image):
        # encoder
        x1 = self.down_conv_1(image)
        x2 = self.max_pooling(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pooling(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pooling(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.max_pooling(x7)
        x9 = self.down_conv_5(x8)
        # decoder
        x = self.up_conv_1(x9)
        x = torch.cat([crop(x7, x), x], 1)
        x = self.down_conv_6(x)
        x = self.up_conv_2(x)
        x = torch.cat([crop(x5, x), x], 1)
        x = self.down_conv_7(x)
        x = self.up_conv_3(x)
        x = torch.cat([crop(x3, x), x], 1)
        x = self.down_conv_8(x)
        x = self.up_conv_4(x)
        x = torch.cat([crop(x1, x), x], 1)
        x = self.down_conv_9(x)
        x = self.conv_1x1(x)
        print(x.size())
        return x


if __name__ == '__main__':
    # test
    fake_input = torch.rand((1, 1, 572, 572))
    model = UNet()
    model(fake_input)