# -*- coding: utf-8 -*-
from torchsummary import summary
import torch
import torch.nn as nn
from collections import OrderedDict
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn.modules.utils import _pair
from timm.models.layers import DropPath, trunc_normal_

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

def proj_1(channel_in, channel_out, kernel, stride, padding):
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

def proj_2(channel_in, channel_out, kernel, stride, padding):
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

class Attention(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 k_v_stride=1,
                 **kwargs
                 ):
        super().__init__()

        self.dim = dim_out
        self.num_heads = num_heads
        self.scale = dim_out ** -0.5

        self.query1 = proj_1(dim_in, dim_out, kernel=3, stride=1, padding=1)
        self.query2 = proj_2(dim_in, dim_out, kernel=3, stride=1, padding=1)  # q,k进行下采样
        self.key1   = proj_1(dim_in, dim_out, kernel=3, stride=k_v_stride, padding=1)
        self.key2   = proj_2(dim_in, dim_out, kernel=3, stride=k_v_stride, padding=1)
        self.value1 = proj_1(dim_in, dim_out, kernel=3, stride=k_v_stride, padding=1)
        self.value2 = proj_2(dim_in, dim_out, kernel=3, stride=k_v_stride, padding=1)

        # self.qkv_encoder = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)
        # self.qkv_decoder = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)

        self.attn_drop_en = nn.Dropout(attn_drop)
        self.attn_drop_de = nn.Dropout(attn_drop)

        self.proj_en = nn.Linear(dim_out, dim_out)
        self.proj_de = nn.Linear(dim_out, dim_out)

        self.proj_drop_en = nn.Dropout(proj_drop)
        self.proj_drop_de = nn.Dropout(proj_drop)
    # decoder/encoder
    def forward(self,x ,y ,h ,w):  # (1,1200,256),(1,1200,256),30,40
        B,N,C = x.shape #1,300,256
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)# (1,256,15,20)
        y = rearrange(y, 'b (h w) c -> b c h w', h=h, w=w)# (1,256,15,20)
        # qkv_encoder = self.qkv_encoder(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q_encoder, k_encoder, v_encoder = qkv_encoder[0], qkv_encoder[1], qkv_encoder[2]  # make torchscript happy (cannot use tensor as tuple)
        #
        # qkv_decoder = self.qkv_decoder(y).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q_decoder, k_decoder, v_decoder = qkv_decoder[0], qkv_decoder[1], qkv_decoder[2]
        q_decoder = self.query1(x).reshape(B,N,self.num_heads,C//self.num_heads).permute(0,2,1,3)#decoder=(1,8,300,32)
        q_encoder = self.query2(y).reshape(B,N,self.num_heads,C//self.num_heads).permute(0,2,1,3)#encoder=(1,8,300,32)

        k_decoder = self.key1(x).reshape(B,N//4,self.num_heads,C//self.num_heads).permute(0,2,1,3)#(1,8,300,32)
        k_encoder = self.key2(y).reshape(B,N//4,self.num_heads,C//self.num_heads).permute(0,2,1,3)#(1,8,300,32)

        v_decoder = self.value1(x).reshape(B,N//4,self.num_heads,C//self.num_heads).permute(0,2,1,3)#(1,8,300,32)
        v_encoder = self.value2(y).reshape(B,N//4,self.num_heads,C//self.num_heads).permute(0,2,1,3)#(1,8,300,32)


        attn_en = (q_decoder @ k_encoder.transpose(-2, -1)) * self.scale #(1,8,300,300)
        attn_en = attn_en.softmax(dim=-1)
        attn_en = self.attn_drop_en(attn_en)#(1,8,300,300)
        x = (attn_en @ v_encoder).transpose(1, 2).reshape(B, N, C)#(1,8,300,32)--->(1,300,8,32)-->(1,300,256)
        x = self.proj_en(x)
        encoder = self.proj_drop_en(x)#encoder=(1,300,256)

        attn_de = (q_encoder @ k_decoder.transpose(-2, -1)) * self.scale#(1,8,300,300)
        attn_de = attn_de.softmax(dim=-1)
        attn_de = self.attn_drop_de(attn_de)
        y = (attn_de @ v_decoder).transpose(1, 2).reshape(B, N, C)#(1,300,256)
        y = self.proj_de(y)
        decoder = self.proj_drop_de(y)#decoder=(1,300,256)
        return encoder, decoder


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
                 k_v_stride=1,
                 **kwargs):
        super().__init__()

        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop, k_v_stride,**kwargs)
        self.drop_path = DropPath(drop_path)
        self.norm2 = norm_layer(dim_in)

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp = Mlp(in_features=dim_out, hidden_features=dim_mlp_hidden, act_layer=act_layer, drop=drop)

    def forward(self, x, y,h,w):  # (1,1200,256)(1,1200,256)
        res_x = x #(1,300,256)
        res_y = y #(1,300,256)

        attn_x, attn_y = self.attn(x, y,h,w)

        x = res_x + self.drop_path(attn_x)  # (1,1200,256)
        y = res_y + self.drop_path(attn_y)  # (1,1200,256)

        x = x + self.drop_path(self.mlp(self.norm1(x)))  # (1,1200,256)
        y = y + self.drop_path(self.mlp(self.norm2(y)))  # (1,1200,256)

        return x, y

class ConvEmbed(nn.Module):

    def __init__(self,
                 in_chans=3,
                 embed_dim=64,
                 patch_size_de=3,
                 patch_size_en=3,
                 stride_de=2,
                 stride_en=2,
                 padding_de=1,
                 padding_en=1,
                 norm_layer=None):
        super().__init__()
        patch_size_de = _pair(patch_size_de)
        patch_size_en = _pair(patch_size_en)
        stride_de = _pair(stride_de)
        stride_en = _pair(stride_en)

        self.proj_de = nn.Conv2d(
            in_chans,
            embed_dim,  # 64
            kernel_size=patch_size_de,  # 3
            stride=stride_de,  # 2
            padding=padding_de  # 1
        )
        self.proj_en = nn.Conv2d(
            in_chans,
            embed_dim,  # 64
            kernel_size=patch_size_en,  # 3
            stride=stride_en,  # 2
            padding=padding_en  # 1
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, y):
        x = self.proj_en(x)  # (1,256,15,20)
        y = self.proj_de(y)  # (1,256,15,20)

        b,c,h,w = x.shape

        embedding_x = rearrange(x, 'b c h w -> b (h w) c')  # (1,300,256)
        embedding_y = rearrange(y, 'b c h w -> b (h w) c')  # (1,300,256)

        embedding_x =self.norm(embedding_x)
        embedding_y = self.norm(embedding_y)

        return embedding_x, embedding_y, h, w

class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x , h, w):
        B, n_patch, hidden = x.size()#1,300,256
        x = x.permute(0, 2, 1)#(1,256,300)
        x = x.contiguous().view(B, hidden, h, w)#(1,256,15,20)
        x = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear', align_corners=True)(x)#(1,256,30,40)
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class SCT(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads=8,
                 embed_dim=256,
                 patch_size_de=3,
                 patch_size_en=3,
                 stride_de=2,
                 stride_en=2,
                 padding_de=1,
                 padding_en=1,
                 in_chans=256,
                 norm_layer=nn.LayerNorm,
                 scale_factor = 2,
                 k_v_stride=1
                 ):
        super().__init__()

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
            norm_layer=nn.LayerNorm,
            k_v_stride=k_v_stride
        )

        self.reconstruct_1 = Reconstruct(in_channels=dim_in,out_channels=dim_out,kernel_size=1,scale_factor=scale_factor)
        self.reconstruct_2 = Reconstruct(in_channels=dim_in,out_channels=dim_out,kernel_size=1,scale_factor=scale_factor)

        self.norm_x = norm_layer(embed_dim)
        self.norm_y = norm_layer(embed_dim)

        self.pos_drop = nn.Dropout(p=0.1)

    # decoder/encoder
    def forward(self, out, fuse):  # (1,256,30,40)

        decoder_x, encoder_y, H, W = self.patch_embed(out,fuse)  # 序列 (1,300,256), (1,300,256),15,20

        x = self.pos_drop(decoder_x)
        y = self.pos_drop(encoder_y)

        x_1, y_1 = self.block(x, y, H, W)  # (1,300,256),(1,300,256)

        final_x = self.reconstruct_1(x_1,H,W)#(1,256,30,40)
        final_y = self.reconstruct_2(y_1,H,W)#(1,256,30,40)

        return final_x, final_y

def main():
    model = SCT(dim_in=256,
                dim_out=256,
                num_heads=8,
                embed_dim=256,
                patch_size_de=3,
                patch_size_en=3,
                stride_de=2,
                stride_en=2,
                padding_de=1,
                padding_en=1,
                in_chans=256,
                norm_layer=nn.LayerNorm)
    model.eval()
    rgb_image = torch.randn(1, 256, 30, 40)
    depth_image = torch.randn(1, 256, 30, 40)
    summary(model, input_size=[(256, 30, 40), (256, 30, 40)], device='cpu')
    with torch.no_grad():
        output_x,output_y = model(rgb_image, depth_image)
    print(output_x.shape,output_y.shape)


if __name__ == '__main__':
    main()