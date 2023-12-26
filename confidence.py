# -*- coding: utf-8 -*-
import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def get_sobel(in_chan, out_chan):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)

    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))  # 自定义修改卷积核的权重
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))

    return sobel_x, sobel_y


def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)  # (1,1,15,20)
    g_y = conv_y(input)  # (1,1,15,20)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2) + 1e-6).clone()
    return g


# class CA(nn.Module):
#     def __init__(self, inc, outc):
#         super(CA, self).__init__()

#         self.sobel_x1_1, self.sobel_y1_1 = get_sobel(inc, outc)

#     def forward(self, boundary_16,boundary_8,boundary_4,boundary):

#         label_extract = run_sobel(self.sobel_x1_1, self.sobel_y1_1, boundary)
#         label_extract_16 = run_sobel(self.sobel_x1_1, self.sobel_y1_1, boundary_16)
#         label_extract_8 = run_sobel(self.sobel_x1_1, self.sobel_y1_1, boundary_8)
#         label_extract_4 = run_sobel(self.sobel_x1_1, self.sobel_y1_1, boundary_4)

#         return label_extract,label_extract_16,label_extract_8,label_extract_4
class CA(nn.Module):
    def __init__(self, inc, outc):
        super(CA, self).__init__()

        self.sobel_x1_1, self.sobel_y1_1 = get_sobel(inc, outc)

    def forward(self, boundary_16,boundary_8,boundary_4,boundary):

        label_extract = run_sobel(self.sobel_x1_1, self.sobel_y1_1, boundary)
        label_extract_16 = run_sobel(self.sobel_x1_1, self.sobel_y1_1, boundary_16)
        label_extract_8 = run_sobel(self.sobel_x1_1, self.sobel_y1_1, boundary_8)
        label_extract_4 = run_sobel(self.sobel_x1_1, self.sobel_y1_1, boundary_4)

        return label_extract,label_extract_16,label_extract_8,label_extract_4