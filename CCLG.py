import logging
import os
from collections import OrderedDict
# from torchsummary import summary
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
class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 **kwargs
                 ):
        super().__init__()

        self.dim = dim_out
        self.num_heads = num_heads
        self.scale = dim_out ** -0.5

        self.query1 = nn.ModuleList()
        self.query2 = nn.ModuleList()
        self.key1 = nn.ModuleList()
        self.key2 = nn.ModuleList()
        self.value1 = nn.ModuleList()
        self.value2 = nn.ModuleList()

        self.softmax = Softmax(dim=3)
        self.attn_dropout = nn.Dropout(0.1)

        self.out1 = nn.Linear(dim_in, dim_out, bias=False)
        self.out2 = nn.Linear(dim_in, dim_out, bias=False)
        self.proj_dropout = nn.Dropout(0.1)

        for _ in range(num_heads):
            query1 = self.proj_1(dim_in, dim_out, kernel=3, stride=2, padding=1)
            query2 = self.proj_2(dim_in, dim_out, kernel=3, stride=2, padding=1)  # q,k进行下采样
            key1 = self.proj_1(dim_in, dim_out, kernel=3, stride=2, padding=1)
            key2 = self.proj_2(dim_in, dim_out, kernel=3, stride=2, padding=1)
            value1 = self.proj_1(dim_in, dim_out, kernel=3, stride=1, padding=1)
            value2 = self.proj_2(dim_in, dim_out, kernel=3, stride=1, padding=1)

            self.query1.append(copy.deepcopy(query1))
            self.query2.append(copy.deepcopy(query2))
            self.key1.append(copy.deepcopy(key1))
            self.key2.append(copy.deepcopy(key2))
            self.value1.append(copy.deepcopy(value1))
            self.value2.append(copy.deepcopy(value2))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def proj_1(self, channel_in, channel_out, kernel, stride, padding):

        proj_1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(
                channel_in,
                channel_out,
                kernel_size=kernel,
                padding=padding,
                stride=stride,
                bias=False,
            )),
            ('bn', nn.BatchNorm2d(channel_out)),
            ('rearrage', Rearrange('b c h w -> b (h w) c')),
        ]))
        return proj_1

    def proj_2(self, channel_in, channel_out, kernel, stride, padding):

        proj_2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(
                channel_in,
                channel_out,
                kernel_size=kernel,
                padding=padding,
                stride=stride,
                bias=False,
            )),
            ('bn', nn.BatchNorm2d(channel_out)),
            ('rearrage', Rearrange('b c h w -> b (h w) c')),
        ]))
        return proj_2
    # decoder/encoder
    def forward(self, x, y, h, w):  # (1,1200,256),(1,1200,256),30,40

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)  # (1,256,30,40)
        y = rearrange(y, 'b (h w) c -> b c h w', h=h, w=w)  # (1,256,30,40)

        multi_head_Q1_list = []
        multi_head_Q2_list = []
        multi_head_K1_list = []
        multi_head_K2_list = []
        multi_head_V1_list = []
        multi_head_V2_list = []

        # x是decoder，y是encoder
        for query1 in self.query1:
            Q1 = query1(x)  # (1,300,256)
            multi_head_Q1_list.append(Q1)  # list{8}

        for key1 in self.key1:
            K1 = key1(y)  # (1,300,256)          #(1,900,256)
            multi_head_K1_list.append(K1)  # (1,300,256)

        for value1 in self.value1:
            V1 = value1(y)  # (1,1200,256)
            multi_head_V1_list.append(V1)  # (1,1200,256)

        # x是decoder，y是encoder
        for query2 in self.query2:
            Q2 = query2(y)  # (1,300,256)
            multi_head_Q2_list.append(Q2)  # list{8} # (1,300,256)

        for key2 in self.key2:
            K2 = key2(x)  # (1,300,256)
            multi_head_K2_list.append(K2)  # (1,300,256)

        for value2 in self.value2:
            V2 = value2(x)  # (1,1200,256)
            multi_head_V2_list.append(V2)  # (1,1200,256)

        multi_head_Q1 = torch.stack(multi_head_Q1_list, dim=1)  # (1,8,300,256)
        multi_head_K1 = torch.stack(multi_head_K1_list, dim=1)  # (1,8,300,256)
        multi_head_V1 = torch.stack(multi_head_V1_list, dim=1)  # (1,8,1200,256)

        multi_head_Q2 = torch.stack(multi_head_Q2_list, dim=1)  # (1,8,300,256)
        multi_head_K2 = torch.stack(multi_head_K2_list, dim=1)  # (1,8,300,256)
        multi_head_V2 = torch.stack(multi_head_V2_list, dim=1)  # (1,8,1200,256)

        multi_head_Q1 = multi_head_Q1.transpose(-1, -2)  # (1,8,256,300)
        multi_head_Q2 = multi_head_Q2.transpose(-1, -2)  # (1,8,256,300)

        attention_scores1 = torch.matmul(multi_head_Q2, multi_head_K1)  # (1,8,256,256)
        attention_scores2 = torch.matmul(multi_head_Q1, multi_head_K2)  # (1,8,256,256)

        attention_scores1 = attention_scores1 * self.scale
        attention_scores2 = attention_scores2 * self.scale

        attention_probs1 = self.softmax(attention_scores1)
        attention_probs2 = self.softmax(attention_scores2)

        attention_probs1 = self.attn_dropout(attention_probs1)  # (1,8,256,256)
        attention_probs2 = self.attn_dropout(attention_probs2)  # (1,8,256,256)

        multi_head_V1 = multi_head_V1.transpose(-1, -2)  # (1,8,256,1200)/encoder
        multi_head_V2 = multi_head_V2.transpose(-1, -2)  # (1,8,256,1200)/decoder

        context_layer1 = torch.matmul(attention_probs1, multi_head_V1)  # (1,8,256,1200)
        context_layer2 = torch.matmul(attention_probs2, multi_head_V2)  # (1,8,256,1200)

        context_layer1 = context_layer1.permute(0, 3, 2, 1).contiguous()  # (1,1200,256,8)
        context_layer2 = context_layer2.permute(0, 3, 2, 1).contiguous()  # (1,1200,256,8)

        context_layer1 = context_layer1.mean(dim=3)  # (1,256,1200)
        context_layer2 = context_layer2.mean(dim=3)  # (1,256,1200)

        o1 = self.out1(context_layer1)  # (1,1200,256)
        o2 = self.out2(context_layer2)  # (1,1200,256)

        o1 = self.proj_dropout(o1)
        o2 = self.proj_dropout(o2)

        return o1, o2 # encoder/decoder


class Block(nn.Module):

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.ReLU,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()

        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.norm2 = norm_layer(dim_in)

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp = Mlp(in_features=dim_out, hidden_features=dim_mlp_hidden, act_layer=act_layer, drop=drop)

    def forward(self, x, y, h, w):  # (1,1200,256)(1,1200,256)
        res_x = x
        res_y = y

        attn_x, attn_y = self.attn(x, y, h, w)

        x = res_x + self.drop_path(attn_x)  # (1,1200,256)
        y = res_y + self.drop_path(attn_y)  # (1,1200,256)

        x = x + self.drop_path(self.mlp(self.norm1(x)))  # (1,1200,256)
        y = y + self.drop_path(self.mlp(self.norm2(y)))  # (1,1200,256)

        return x, y


class ConvEmbed(nn.Module):

    def __init__(self,
                 in_chans=3,
                 embed_dim=64,
                 patch_size_de=7,
                 patch_size_en=7,
                 stride_de=4,
                 stride_en=4,
                 padding_de=2,
                 padding_en=2,
                 norm_layer=None):
        super().__init__()
        patch_size_de = _pair(patch_size_de)
        patch_size_en = _pair(patch_size_en)
        stride_de = _pair(stride_de)
        stride_en = _pair(stride_en)

        self.proj_de = nn.Conv2d(
            in_chans,
            embed_dim,  # 64
            kernel_size=patch_size_de,  # 7
            stride=stride_de,  # 4
            padding=padding_de  # 2
        )
        self.proj_en = nn.Conv2d(
            in_chans,
            embed_dim,  # 64
            kernel_size=patch_size_en,  # 7
            stride=stride_en,  # 4
            padding=padding_en  # 2
        )

    def forward(self, x, y):
        x = self.proj_en(x)  # (1,256,30,40)
        y = self.proj_de(y)  # (1,256,30,40)
        embedding_x = rearrange(x, 'b c h w -> b (h w) c')  # (1,1200,256)
        embedding_y = rearrange(y, 'b c h w -> b (h w) c')  # (1,1200,256)

        return embedding_x, embedding_y


class CCT(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads=8,
                 embed_dim=256,
                 patch_size_de=7,
                 patch_size_en=7,
                 stride_de=4,
                 stride_en=4,
                 padding_de=2,
                 padding_en=2,
                 in_chans=512,
                 norm_layer=nn.LayerNorm,
                 ):
        super(CCT, self).__init__()

        self.patch_embed = ConvEmbed(
            patch_size_de=patch_size_de,
            patch_size_en=patch_size_en,
            in_chans=in_chans,
            stride_de=stride_de,
            stride_en=stride_en,
            padding_de=padding_de,
            padding_en=padding_en,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )
        self.block = Block(
            dim_in,
            dim_out,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.ReLU,
            norm_layer=nn.LayerNorm
        )
        self.norm_x = norm_layer(embed_dim)
        self.norm_y = norm_layer(embed_dim)

        self.pos_drop = nn.Dropout(p=0.1)

    # decoder/encoder
    def forward(self, out, fuse):  # (1,256,30,40)
        B, C, H, W = out.shape
        decoder, encoder = self.patch_embed(out, fuse)  # 序列 (1,1200,256)

        x_decoder = self.pos_drop(decoder)
        y_encoder = self.pos_drop(encoder)

        x_1, y_1 = self.block(x_decoder, y_encoder, H, W)  # (1,1200,256)

        x = self.norm_x(x_1)  # (1,1200,256)
        y = self.norm_y(y_1)  # (1,1200,256)

        final_x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)  # (1,256,30,40)
        final_y = rearrange(y, 'b (h w) c -> b c h w', h=H, w=W)  # (1,256,30,40)

        return final_x, final_y
