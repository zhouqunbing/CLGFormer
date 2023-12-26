import logging
import os
from collections import OrderedDict
from torchsummary import summary
import numpy as np
import scipy
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn.modules.utils import _pair
from timm.models.layers import DropPath, trunc_normal_
from torch.nn import Softmax, Conv2d, LayerNorm
from src.models.resnet import ResNet34
import warnings
from DSF import FeatureRectifyModule
from CLGT import CCT
from SCLGT import SCT
from confidence import CA


def convt_bn_relu(ch_in, ch_out, kernel=3, stride=2, padding=1, output_padding=1,
                  bn=True, relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel, stride, padding,
                                     output_padding, bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    layers = nn.Sequential(*layers)
    return layers


def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True, relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding, bias=not bn))  # 如果卷积层后面有bn的话，就禁用bias。
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    layers = nn.Sequential(*layers)
    return layers


class CCA(nn.Module):
    def __init__(self, inchannel):  # (256,256)
        super().__init__()
        reduction = 16
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Sequential(
            nn.Conv2d(inchannel, inchannel // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel // reduction, inchannel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, g):  # fuse,out = (1,256,30,40)
        weighting_g = F.adaptive_avg_pool2d(g, 1)  # (1,256,1,1)
        weighting_g = self.fc(weighting_g)  # (1,256,1,1)
        g_after_channel = g * weighting_g  # (1,256,30,40)
        fuse = self.relu(g_after_channel)
        return fuse

class spatial_attention(nn.Module):
    def __init__(self, in_channel,reduction=4):
        super(spatial_attention, self).__init__()
        self.mlp_1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1),
            nn.BatchNorm2d(in_channel // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // reduction,in_channel,kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_1 = self.mlp_1(x)
        out = x_1 + x
        return out


class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels):  # (256,128)/(128,64)
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.coatt_1 = CCA(in_channels)
        # self.coatt_2 = spatial_attention(in_channels)
        self.coatt_2 = CCA(in_channels)

        self.upsample = convt_bn_relu(in_channels, out_channels)

        self.inter1 = conv_bn_relu(in_channels * 2, in_channels, kernel=3, stride=1, padding=1)
        self.inter2 = conv_bn_relu(in_channels, in_channels, kernel=3, stride=1, padding=1)
        self.inter3 = conv_bn_relu(in_channels, in_channels, kernel=3, stride=1, padding=1)

    def forward(self, fuse, out, fuse_a, out_a):  # (1,256,30,40),(1,512,30,40),(1,512,30,40)(1,256,30,40)

        out = torch.cat([fuse, out], dim=1)  # (1,512,30,40)
        out1 = self.inter1(out)  # (1,256,30,40)

        out2 = self.coatt_1(out1)  # (1,256,30,40)
        out2 = out2 + fuse_a
        out2 = self.inter2(out2)  # (1,256,30,40)

        out3 = out1 + out2

        out4 = self.coatt_2(out3)  # (1,256,30,40)
        out4 = out4 + out_a
        out4 = self.inter3(out4)  # (1,256,30,40)

        out5 = out3 + out4

        x = self.upsample(out5)  ##(1,256,60,80)

        return x

class CLGFormer(nn.Module):
    def __init__(self,
                 num_classes=40,
                 encoder_rgb='resnet34',
                 encoder_depth='resnet34',
                 encoder_block='BasicBlock',
                 pretrained_on_imagenet=True,
                 fuse_depth_in_rgb_encoder='FeatureRectifyModule',
                 pretrained_dir='',
                 activation='relu',
                 ):
        super(CLGFormer, self).__init__()
        in_channels = 64

        # set activation function
        if activation.lower() == 'relu':
            self.activation = nn.ReLU(inplace=True)

        else:
            raise NotImplementedError(
                'Only relu, swish and hswish as activation function are '
                'supported so far. Got {}'.format(activation))

        if encoder_rgb == 'resnet50' or encoder_depth == 'resnet50':
            warnings.warn('Parameter encoder_block is ignored for ResNet50. '
                          'ResNet50 always uses Bottleneck')

        # rgb encoder
        if encoder_rgb == 'resnet34':
            self.encoder_rgb = ResNet34(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation)

        else:
            raise NotImplementedError(
                'Only ResNets are supported for '
                'encoder_rgb. Got {}'.format(encoder_rgb))

        # depth encoder
        if encoder_depth == 'resnet34':
            self.encoder_depth = ResNet34(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation,
                input_channels=1)
        else:
            raise NotImplementedError(
                'Only ResNets are supported for '
                'encoder_depth. Got {}'.format(encoder_rgb))

        if fuse_depth_in_rgb_encoder == 'FeatureRectifyModule':
            self.se_layer0 = FeatureRectifyModule(
                64)
            self.se_layer1 = FeatureRectifyModule(
                self.encoder_rgb.down_4_channels_out,
            )
            self.se_layer2 = FeatureRectifyModule(
                self.encoder_rgb.down_8_channels_out,
            )
            self.se_layer3 = FeatureRectifyModule(
                self.encoder_rgb.down_16_channels_out,
            )
            self.se_layer4 = FeatureRectifyModule(
                self.encoder_rgb.down_32_channels_out,
            )

        self.trans1 = UpBlock_attention(in_channels * 4, in_channels * 2)
        self.trans2 = UpBlock_attention(in_channels * 2, in_channels)
        self.trans3 = UpBlock_attention(in_channels, in_channels)

        self.edge_extract = CA(num_classes,1)

        self.up1 = convt_bn_relu(in_channels * 8, in_channels * 4, kernel=3, stride=2, output_padding=1)
        self.up5 = convt_bn_relu(in_channels, in_channels, kernel=3, stride=2, output_padding=1)

        self.final = nn.Conv2d(in_channels, num_classes, kernel_size=1, padding=0)

        self.transfusion1 = CCT(dim_in=256, dim_out=256, num_heads=8, embed_dim=256, patch_size_de=7, patch_size_en=7,
                                stride_de=1, stride_en=1, padding_de=3, padding_en=3, in_chans=256,
                                norm_layer=nn.LayerNorm)

        self.sct1 = SCT(dim_in=256, dim_out=256, num_heads=8, embed_dim=256, patch_size_de=3, patch_size_en=3,
                        stride_de=1, stride_en=1, padding_de=1, padding_en=1, in_chans=256, norm_layer=nn.LayerNorm,
                        k_v_stride=2, scale_factor=1)

        self.transfusion2 = CCT(dim_in=128, dim_out=128, num_heads=4, embed_dim=128, patch_size_de=5, patch_size_en=5,
                                stride_de=1, stride_en=1, padding_de=2, padding_en=2, in_chans=128,
                                norm_layer=nn.LayerNorm)

        self.sct2 = SCT(dim_in=128, dim_out=128, num_heads=4, embed_dim=128, patch_size_de=3, patch_size_en=3,
                        stride_de=2, stride_en=2, padding_de=1, padding_en=1, in_chans=128, norm_layer=nn.LayerNorm,
                        k_v_stride=2, scale_factor=2)

        self.transfusion3 = CCT(dim_in=64, dim_out=64, num_heads=2, embed_dim=64, patch_size_de=3, patch_size_en=3,
                                stride_de=1, stride_en=1, padding_de=1, padding_en=1, in_chans=64,
                                norm_layer=nn.LayerNorm)

        self.sct3 = SCT(dim_in=64, dim_out=64, num_heads=2, embed_dim=64, patch_size_de=3, patch_size_en=3,
                        stride_de=2, stride_en=2, padding_de=1, padding_en=1, in_chans=64, norm_layer=nn.LayerNorm,
                        k_v_stride=2, scale_factor=2)

        self.out_side16 = nn.Conv2d(in_channels * 4, num_classes, kernel_size=3, padding=1)
        self.side_out8 = nn.Conv2d(in_channels * 2, num_classes, kernel_size=3, padding=1)
        self.side_out4 = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.side_out2 = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)

    def forward(self, rgb,depth):  # rgb, depth
        b, c, h1, w1 = rgb.shape

        rgb = self.encoder_rgb.forward_first_conv(rgb)  # (1,3,480,640)--->(1,64,240,320)
        depth = self.encoder_depth.forward_first_conv(depth)  # (1,3,480,640)--->(1,64,240,320)
        fuse = self.se_layer0(rgb, depth)  # (1,64,240,320)

        rgb = F.max_pool2d(fuse, kernel_size=3, stride=2, padding=1)  # (1,64,120,160)
        depth = F.max_pool2d(depth, kernel_size=3, stride=2, padding=1)  # (1,64,120,160)

        # block 1
        rgb = self.encoder_rgb.forward_layer1(rgb)  # (1,64,120,160)
        depth = self.encoder_depth.forward_layer1(depth)  # (1,64,120,160)
        fuse1 = self.se_layer1(rgb, depth)  ##(1,64,120,160)

        # block 2
        rgb = self.encoder_rgb.forward_layer2(fuse1)  # (1,128,60,80)
        depth = self.encoder_depth.forward_layer2(depth)  # (1,128,60,80)
        fuse2 = self.se_layer2(rgb, depth)  ##(1,128,60,80)

        # block 3
        rgb = self.encoder_rgb.forward_layer3(fuse2)  # (1,256,30,40)
        depth = self.encoder_depth.forward_layer3(depth)  # (1,256,30,40)
        fuse3 = self.se_layer3(rgb, depth)  # (1,256,30,40)

        # block 4
        rgb = self.encoder_rgb.forward_layer4(fuse3)  # (1,512,30,40)
        depth = self.encoder_depth.forward_layer4(depth)  # (1,512,30,40)
        fuse4 = self.se_layer4(rgb, depth)  # (1,256,30,40)

        fuse5 = self.up1(fuse4)  # (1,256,30,40)

        out_side16 = self.out_side16(fuse5)  # (1,40,30,40)
        segmap_16 = F.interpolate(out_side16, size=(h1, w1), mode='bilinear', align_corners=False)  # (1,40,480,640)

        # transformer fusion/(decoder,encoder)
        out_en1, out_de1 = self.transfusion1(fuse5,
                                             fuse3)  # encoder/decoder=((1,256,30,40),(1,256,30,40))--->(1,256,30,40)
        out_en2, out_de2 = self.sct1(fuse5,
                                     fuse3)  # encoder/decoder=(1,256,30,40),(1,256,30,40)--->(1,256,30,40),(1,256,30,40)
        fused_1 = self.trans1(out_en1, out_de1, out_en2, out_de2)  # (1,128,60,80)

        out_side8 = self.side_out8(fused_1)  # (1,40,60,80)
        segmap_8 = F.interpolate(out_side8, size=(h1, w1), mode='bilinear', align_corners=False)  # (1,40,480,640)

        out_en2, out_de2 = self.transfusion2(fused_1,fuse2)  # (1,128,60,80),(1,128,60,80)--->(1,128,60,80),(1,128,60,80)
        out_en3, out_de3 = self.sct2(fused_1, fuse2)  # (1,128,60,80),(1,128,60,80)--->(1,128,60,80),(1,128,60,80)
        fused_2 = self.trans2(out_en2, out_de2, out_en3, out_de3)  # (1,64,120,160)

        out_side4 = self.side_out4(fused_2)  # (1,40,120,160)
        segmap_4 = F.interpolate(out_side4, size=(h1, w1), mode='bilinear', align_corners=False)  # (1,40,480,640)

        out_en4, out_de4 = self.transfusion3(fused_2,fuse1)  # (1,64,120,160),(1,64,120,160)--->(1,64,120,160),(1,64,120,160)
        out_en5, out_de5 = self.sct3(fused_2, fuse1)  # (1,64,120,160),(1,64,120,160)--->(1,64,120,160),(1,64,120,160)
        fused_3 = self.trans3(out_en4, out_de4, out_en5, out_de5)  # (1,64,240,320)

        # decoder
        fused4 = self.up5(fused_3)  # (1,64,480,640)
        final_out = self.final(fused4)  # (1,40,480,640)
        edge_final_out,edge_out_16, edge_out_8, edge_out_4 = self.edge_extract(final_out,segmap_16, segmap_8, segmap_4)

        if self.training:
            return final_out, segmap_16, segmap_8, segmap_4, \
                   edge_final_out, edge_out_16, edge_out_8, edge_out_4

        return final_out


def main():
    # --------------------------------实例化CLGFormer-------------------------
    model = CLGFormer()  # (传入参数)
    model.eval()
    rgb = torch.rand(1, 3, 480, 640)
    depth = torch.rand(1, 1, 480, 640)
    summary(model, input_size=[(3, 480, 640), (1, 480, 640)], device='cpu')

    with torch.no_grad():
        output = model(rgb,depth)
    print(output.shape)


if __name__ == '__main__':
    main()
