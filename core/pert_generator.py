from turtle import forward
import torch
from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ngf = 64

class PertGenerator(nn.Module):
    def __init__(self, config):
        super(PertGenerator, self).__init__()

        self.inception = config.inception
        self.epsilon = config.epsilon

        self.mode = config.mode
        
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(True)
        )

        # Input size = 3, n, n
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(True)
        )

        # Input size = 3, n/4, n/4
        # Residual Blocks: 7
        residual_blocks = [ResidualBlock(ngf * 4) for i in range(7)]
        self.resblock = nn.Sequential(
            *residual_blocks
        )

        # Input size = 3, n/4, n/4
        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(True)
        )

        # Input size = 3, n, n
        self.blockf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0)
        )

        # self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)

        self.initialize_weights()

    def forward(self, images):
        x = self.block1(images)
        x = self.block2(x)
        x = self.block3(x)
        x = self.resblock(x)
        x = self.upsampl1(x)
        x = self.upsampl2(x)
        x = self.blockf(x)
        # if self.inception:
        #     x = self.crop(x)
        #perbtution limit [0, 1]
        # x = torch.clamp(torch.tanh(x), min=-self.epsilon, max=self.epsilon)
        pert = (torch.sigmoid(x) -0.5) * 2 * self.epsilon
        images_with_pert = torch.clamp(images + pert, min=-1, max=1) 
        return images_with_pert - images

    def initialize_weights(self):
        for m in self.modules():        
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.3)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1) 		 
                m.bias.data.zero_()



class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(True),

            nn.Dropout(0.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters)
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual



if __name__ == '__main__':
    netG = PertGenerator()
    test_sample = torch.rand(1, 3, 32, 32)
    print('Generator output:', netG(test_sample).size())
    print('Generator parameters:', sum(p.numel() for p in netG.parameters() if p.requires_grad))
    