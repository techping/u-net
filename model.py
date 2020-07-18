# File: model.py
# U-Net Model Implementation
# Ziping Chen

import torch
import torch.nn as nn
import torch.nn.functional as F

def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3),
        nn.BatchNorm2d(out_c),
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
    diff_h = (in_tensor.size()[2] - out_tensor.size()[2])
    left_h = diff_h // 2
    diff_w = (in_tensor.size()[3] - out_tensor.size()[3])
    left_w = diff_w // 2
    return in_tensor[:, :, left_h:-(diff_h-left_h), left_w:-(diff_w-left_w)]

def pad(in_tensor, out_tensor):
    diff_h = (in_tensor.size()[2] - out_tensor.size()[2])
    left_h = diff_h // 2
    diff_w = (in_tensor.size()[3] - out_tensor.size()[3])
    left_w = diff_w // 2
    return F.pad(out_tensor, [left_w, diff_w - left_w,
                                left_h, diff_h - left_h])


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.contracting_path()
        self.expansive_path()

    def contracting_path(self):
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(self.in_channels, 64)
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
        self.conv_1x1 = nn.Conv2d(in_channels=64, out_channels=self.num_classes, kernel_size=1)

    def forward(self, image):
        input_shape = image.shape[2:]
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
        x = torch.cat([x7, pad(x7, x)], 1)
        x = self.down_conv_6(x)
        x = self.up_conv_2(x)
        x = torch.cat([x5, pad(x5, x)], 1)
        x = self.down_conv_7(x)
        x = self.up_conv_3(x)
        x = torch.cat([x3, pad(x3, x)], 1)
        x = self.down_conv_8(x)
        x = self.up_conv_4(x)
        x = torch.cat([x1, pad(x1, x)], 1)
        x = self.down_conv_9(x)
        x = self.conv_1x1(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x


if __name__ == '__main__':
    # from torchviz import make_dot, make_dot_from_trace

    # # test
    # fake_input = torch.rand((1, 3, 572, 572))
    # model = UNet(3, 1)
    # # output = model(fake_input)
    # # print(output.shape)

    # dot = make_dot(model(fake_input), params=dict(model.named_parameters()))
    # dot.format = 'png'
    # dot.render("old")
    pass