import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm
from torchvision import models, utils

from arcface.iresnet import *
import sys
sys.path.append('pixel2style2pixel/')
from pixel2style2pixel.models.stylegan2.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class fs_encoder_v2(nn.Module):
    def __init__(self, n_styles=18, opts=None, residual=False, use_coeff=False, resnet_layer=None, video_input=False, f_maps=512, stride=(1, 1)):
        super(fs_encoder_v2, self).__init__()  

        resnet50 = iresnet50()
        resnet50.load_state_dict(torch.load(opts.arcface_model_path))

        # input conv layer
        if video_input:
            self.conv = nn.Sequential(
                nn.Conv2d(6, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                *list(resnet50.children())[1:3]
            )
        else:
            self.conv = nn.Sequential(*list(resnet50.children())[:3])
        
        # define layers
        self.block_1 = list(resnet50.children())[3] # 15-18
        self.block_2 = list(resnet50.children())[4] # 10-14
        self.block_3 = list(resnet50.children())[5] # 5-9
        self.block_4 = list(resnet50.children())[6] # 1-4
        self.content_layer = nn.Sequential(
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.PReLU(num_parameters=512),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((3,3))
        self.styles = nn.ModuleList()
        for i in range(n_styles):
            self.styles.append(nn.Linear(960 * 9, 512))

        # self.input_layer = nn.Linear(1000, 3*256*256)
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())
        #--------------CelebA--------------------------------------------------------
        self.l1 = nn.Sequential(
            nn.Linear(1000, 64 * 8 * 8 * 8, bias=False),
            nn.BatchNorm1d(64 * 8 * 8 * 8),
            nn.ReLU())
        self.l2 = nn.Sequential(
            dconv_bn_relu(64 * 8, 64 * 4),
            dconv_bn_relu(64 * 4, 64 * 2),
            dconv_bn_relu(64 * 2, 64),
            dconv_bn_relu(64, 32),
            nn.ConvTranspose2d(32, 3, 5, 2, padding=2, output_padding=1),
            nn.Sigmoid())
        # ----------------------------------------------------------------------------
        # --------------Pubfig--------------------------------------------------------
        # self.l1 = nn.Sequential(
        #     nn.Linear(50, 32 * 4 * 4 * 4, bias=False),
        #     nn.BatchNorm1d(32 * 4 * 4 * 4),
        #     nn.ReLU())
        # self.l2 = nn.Sequential(
        #     dconv_bn_relu(32, 16),
        #     dconv_bn_relu(16, 8),
        #     dconv_bn_relu(8, 4),
        #     dconv_bn_relu(4, 2),
        #     nn.ConvTranspose2d(2, 3, 5, 2, padding=2, output_padding=1),
        #     nn.Sigmoid())
        # ----------------------------------------------------------------------------
        # --------------Facescrub-----------------------------------------------------
        # self.l1 = nn.Sequential(
        #     nn.Linear(200, 32 * 8 * 8 * 4, bias=False),
        #     nn.BatchNorm1d(32 * 8 * 8 * 4),
        #     nn.ReLU())
        # self.l2 = nn.Sequential(
        #     dconv_bn_relu(128, 64),
        #     dconv_bn_relu(64, 32),
        #     dconv_bn_relu(32, 16),
        #     dconv_bn_relu(16, 8),
        #     nn.ConvTranspose2d(8, 3, 5, 2, padding=2, output_padding=1),
        #     nn.Sigmoid())
        # ----------------------------------------------------------------------------

        # self.l1 = nn.Sequential(nn.Linear(1000, 3 * 128 * 128),
        #                         nn.BatchNorm1d(3 * 128 * 128),
        #                         nn.ReLU(),
        #                         nn.Linear(3 * 128 * 128, 3 * 256 * 256),
        #                         nn.BatchNorm1d(3 * 256 * 256),
        #                         nn.ReLU()
        #                         )


    def forward(self, x):
        latents = []
        features = []
        x = x.squeeze(1)
        #------------------------------
        y = self.l1(x)
        y = y.view(y.size(0), -1, 8, 8)
        x = self.l2(y)
        # ------------------------------
        x = x.view(x.size(0), -1, 256, 256)
        outimg = x #torch.Size([1, 3, 256, 256])
        x = self.conv(x) #torch.Size([1, 64, 256, 256])
        x = self.block_1(x) #torch.Size([1, 64, 128, 128])
        features.append(self.avg_pool(x)) #torch.Size([1, 64, 3, 3])
        x = self.block_2(x) #torch.Size([1, 128, 64, 64])
        features.append(self.avg_pool(x)) #torch.Size([1, 128, 3, 3])
        x = self.block_3(x) #torch.Size([1, 256, 32, 32])
        content = self.content_layer(x) #torch.Size([1, 512, 16, 16])
        features.append(self.avg_pool(x)) #torch.Size([1, 256, 3, 3])
        x = self.block_4(x) #torch.Size([1, 512, 16, 16])
        features.append(self.avg_pool(x)) #torch.Size([1, 512, 3, 3])
        x = torch.cat(features, dim=1)
        x = x.view(x.size(0), -1) #torch.Size([1, 8640])
        for i in range(len(self.styles)):
            latents.append(self.styles[i](x))
        out = torch.stack(latents, dim=1) #torch.Size([1, 18, 512])
        return out, content, outimg


class fs_encoder_v2_fix(nn.Module):
    def __init__(self, n_styles=18, opts=None, residual=False, use_coeff=False, resnet_layer=None, video_input=False,
                 f_maps=512, stride=(1, 1)):
        super(fs_encoder_v2_fix, self).__init__()

        resnet50 = iresnet50()
        resnet50.load_state_dict(torch.load(opts.arcface_model_path))

        # input conv layer
        if video_input:
            self.conv = nn.Sequential(
                nn.Conv2d(6, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                *list(resnet50.children())[1:3]
            )
        else:
            self.conv = nn.Sequential(*list(resnet50.children())[:3])

        # define layers
        self.block_1 = list(resnet50.children())[3]  # 15-18
        self.block_2 = list(resnet50.children())[4]  # 10-14
        self.block_3 = list(resnet50.children())[5]  # 5-9
        self.block_4 = list(resnet50.children())[6]  # 1-4
        self.content_layer = nn.Sequential(
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.PReLU(num_parameters=512),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((3, 3))
        self.styles = nn.ModuleList()
        for i in range(n_styles):
            self.styles.append(nn.Linear(960 * 9, 512))

    def forward(self, x):
        latents = []
        features = []
        x = self.conv(x)
        x = self.block_1(x)
        features.append(self.avg_pool(x))
        x = self.block_2(x)
        features.append(self.avg_pool(x))
        x = self.block_3(x)
        content = self.content_layer(x)
        features.append(self.avg_pool(x))
        x = self.block_4(x)
        features.append(self.avg_pool(x))
        x = torch.cat(features, dim=1)
        x = x.view(x.size(0), -1)
        for i in range(len(self.styles)):
            latents.append(self.styles[i](x))
        out = torch.stack(latents, dim=1)
        return out, content



class fs_encoder_v3(nn.Module):
    def __init__(self, n_styles=18):
        super(fs_encoder_v3, self).__init__()


        self.styles = nn.ModuleList()
        for i in range(n_styles):
            self.styles.append(nn.Sequential(PixelNorm(),
                                nn.Linear(1000, 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU()))

    def forward(self, x):
        latents = []
        features = []
        x = x.squeeze(1)

        for i in range(len(self.styles)):
            latents.append(self.styles[i](x))

        out = torch.stack(latents, dim=1)   # torch.Size([1, 18, 512])
        # print(out[0][0])
        return out


class fs_encoder_v4(nn.Module):
    def __init__(self):
        super(fs_encoder_v4, self).__init__()

        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())
        self.l0 = nn.Sequential(
            nn.Linear(1000, 100, bias=False),
            nn.BatchNorm1d(100),
            nn.ReLU())
        self.l1 = nn.Sequential(
            nn.Linear(100, 64 * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(64 * 8 * 4 * 4),
            nn.ReLU())
        self.l2 = nn.Sequential(
            dconv_bn_relu(64 * 8, 64 * 4),
            dconv_bn_relu(64 * 4, 64 * 2),
            dconv_bn_relu(64 * 2, 64),
            nn.ConvTranspose2d(64, 3, 5, 2, padding=2, output_padding=1),
            # dconv_bn_relu(64, 32),
            # nn.ConvTranspose2d(32, 3, 5, 2, padding=2, output_padding=1),
            nn.Sigmoid())

    def forward(self, x):
        latents = []
        features = []
        x = x.squeeze(1)
        # print(x.shape)
        # x = self.input_layer(x)
        # ------------------------------
        x = self.l0(x)
        y = self.l1(x)
        # print(y.shape)
        y = y.view(y.size(0), -1, 8, 8)
        # print(y.shape)
        x = self.l2(y)
        # print(x.shape)
        # ------------------------------
        # x = self.l1(x)
        # print(x.shape)
        # x = x.view(x.size(0), -1, 256, 256)
        x = x.view(x.size(0), -1, 64, 64)
        outimg = x

        return outimg

class fs_encoder_v5(nn.Module):
    def __init__(self):
        super(fs_encoder_v5, self).__init__()


        self.l0 = nn.Sequential(
            nn.Linear(1000, 100, bias=False),
            nn.BatchNorm1d(100),
            nn.ReLU())


    def forward(self, x):
        latents = []
        features = []
        x = x.squeeze(1)
        # print(x.shape)
        # x = self.input_layer(x)
        # ------------------------------
        x = self.l0(x)

        outimg = x

        return outimg




