import torch
import torch.nn as nn


class FilterLayer(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):  # 128,64
        super(FilterLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, out_planes // reduction),  # (128,4)
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // reduction, out_planes * 2),  # (4,128)
            nn.Sigmoid()
        )
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _ = x.size()  # (1,128,480,640)
        y = self.avg_pool(x).view(b, c)  # (1,128)
        y = self.fc(y).view(b, self.out_planes * 2, 1)  # (1,128)--->(1,128,1)
        y = y.reshape(b, 2, self.out_planes, 1, 1).permute(1, 0, 2, 3, 4)  # (1,128,1)>(1,2,64,1,1)>(2,1,64,1,1)
        return y


class FeatureRectifyModule(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(FeatureRectifyModule, self).__init__()

        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 1, 1, bias=False),
                                     nn.BatchNorm2d(in_channel), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 4, 4, bias=False),
                                     nn.BatchNorm2d(in_channel), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 8, 8, bias=False),
                                     nn.BatchNorm2d(in_channel), nn.LeakyReLU(0.1, inplace=True))

        self.mlp_1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // reduction, 3, kernel_size=1),
            nn.Sigmoid(),
        )
        self.split1 = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.split2 = nn.Conv2d(in_channel, in_channel, kernel_size=1)

        self.filter = FilterLayer(2 * in_channel, in_channel, reduction)  # (128,64)

    def forward(self, x, y):  # (1,64,480,640),(1,64,480,640)
        B, C, H, W = x.shape

        small = self.conv1_1(x)  # (1,64,480,640)
        medium = self.conv2_1(x)  # (1,64,480,640)
        large = self.conv3_1(x)  # (1,64,480,640)

        conf = self.mlp_1(y).reshape(B, 3, 1, H, W).permute(1, 0, 2, 3, 4)  # (3,1,1,480,640)
        out = conf[0] * small + conf[1] * medium + conf[2] * large  # (1,64,480,640)

        rgb_split = self.split1(out)  # (1,64,480,640)
        depth_split = self.split2(out)  # (1,64,480,640)

        rgb_out = x + rgb_split
        depth_out = y + depth_split

        combine = torch.cat((x, y), dim=1)  # (1,128,480,640)
        channel_weight = self.filter(combine)  # (2,1,64,1,1)

        out_1 = channel_weight[0] * x + depth_out  # (1,64,480,640)
        out_2 = channel_weight[1] * y + rgb_out  # (1,64,480,640)

        out = out_1 + out_2  # (1,64,480,640)

        return out
