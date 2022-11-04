"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
from torchvision import transforms

from typing import Iterable
import numpy as np


class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
        @notice: avoid in-place ops.
        https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x)  # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp


class ApplyStyle(nn.Module):
    """
    @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """

    def __init__(self, latent_size, channels):
        super(ApplyStyle, self).__init__()
        self.linear = nn.Linear(latent_size, channels * 2)

    def forward(self, x, latent):
        style = self.linear(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        # x = x * (style[:, 0] + 1.) + style[:, 1]
        x = x * (style[:, 0] * 1 + 1.0) + style[:, 1] * 1
        return x


class ResnetBlock_Adain(nn.Module):
    def __init__(self, dim, latent_size, padding_type, activation=nn.ReLU(True)):
        super(ResnetBlock_Adain, self).__init__()

        p = 0
        conv1 = []
        if padding_type == "reflect":
            conv1 += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv1 += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv1 += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), InstanceNorm()]
        self.conv1 = nn.Sequential(*conv1)
        self.style1 = ApplyStyle(latent_size, dim)
        self.act1 = activation

        p = 0
        conv2 = []
        if padding_type == "reflect":
            conv2 += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv2 += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv2 += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), InstanceNorm()]
        self.conv2 = nn.Sequential(*conv2)
        self.style2 = ApplyStyle(latent_size, dim)

    def forward(self, x, dlatents_in_slice):
        y = self.conv1(x)
        y = self.style1(y, dlatents_in_slice)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.style2(y, dlatents_in_slice)
        out = x + y
        return out


class Generator_Adain_Upsample(nn.Module):
    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        latent_size: int,
        n_blocks: int = 6,
        deep: bool = False,
        use_last_act: bool = True,
        norm_layer: torch.nn.Module = nn.BatchNorm2d,
        padding_type: str = "reflect",
    ):
        assert n_blocks >= 0
        super(Generator_Adain_Upsample, self).__init__()

        activation = nn.ReLU(True)

        self.deep = deep
        self.use_last_act = use_last_act

        self.to_tensor_normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.to_tensor = transforms.Compose([transforms.ToTensor()])

        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.first_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, kernel_size=7, padding=0),
            norm_layer(64),
            activation,
        )
        # downsample
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            norm_layer(128),
            activation,
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            norm_layer(256),
            activation,
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            norm_layer(512),
            activation,
        )

        if self.deep:
            self.down4 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                norm_layer(512),
                activation,
            )

        # resnet blocks
        BN = []
        for i in range(n_blocks):
            BN += [
                ResnetBlock_Adain(
                    512,
                    latent_size=latent_size,
                    padding_type=padding_type,
                    activation=activation,
                )
            ]
        self.BottleNeck = nn.Sequential(*BN)

        if self.deep:
            self.up4 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                activation,
            )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            activation,
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            activation,
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            activation,
        )
        if self.use_last_act:
            self.last_layer = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
                torch.nn.Tanh(),
            )
        else:
            self.last_layer = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
            )

    def to(self, device):
        super().to(device)
        self.device = device
        self.imagenet_mean = self.imagenet_mean.to(device)
        self.imagenet_std = self.imagenet_std.to(device)
        return self

    def forward(self, x: Iterable[np.ndarray], dlatents: torch.Tensor, need_trans = True):
        if need_trans:
            if self.use_last_act:
                x = [self.to_tensor(_) for _ in x]
                x = torch.stack(x, dim=0)
            else:
                x = [self.to_tensor_normalize(_) for _ in x]
                x = torch.stack(x, dim=0)

        x = x.to(self.device)

        skip1 = self.first_layer(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        if self.deep:
            skip4 = self.down3(skip3)
            x = self.down4(skip4)
        else:
            x = self.down3(skip3)

        for i in range(len(self.BottleNeck)):
            x = self.BottleNeck[i](x, dlatents)

        if self.deep:
            x = self.up4(x)

        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        x = self.last_layer(x)

        if self.use_last_act:
            x = (x + 1) / 2
        else:
            x = x * self.imagenet_std + self.imagenet_mean

        return x
