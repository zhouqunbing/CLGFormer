# -*- coding: utf-8 -*-
import os
import sys

from torch.nn import functional as F
import pandas as pd
import numpy as np
from torch import nn
import torch
import torch
from torch.autograd import Variable


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, device, weight):
        super(CrossEntropyLoss2d, self).__init__()
        self.weight = torch.tensor(weight).to(device)
        self.num_classes = len(self.weight) + 1  # +1 for void
        if self.num_classes < 2**8:
            self.dtype = torch.uint8
        else:
            self.dtype = torch.int16
        self.ce_loss = nn.CrossEntropyLoss(
            torch.from_numpy(np.array(weight)).float(),
            reduction='none',
            ignore_index=-1
        )
        self.ce_loss.to(device)

    def forward(self, inputs_scales, targets_scales):#(包含深监督的输出)
        losses = []

        for inputs, targets in zip(inputs_scales, targets_scales):
            # mask = targets > 0
            targets_m_1 = targets.clone() #深拷贝
            targets_m = targets_m_1-1 #减去
            # targets_to_one_hot = torch.nn.functional.one_hot(targets_m.to(torch.int64)) #值为-1的样本不参与计算
            loss_all = self.ce_loss(inputs, targets_m.long())

            number_of_pixels_per_class = \
                torch.bincount(targets.flatten().type(self.dtype),
                               minlength=self.num_classes)
            divisor_weighted_pixel_sum = \
                torch.sum(number_of_pixels_per_class[1:] * self.weight)   # without void

            losses.append(torch.sum(loss_all) / divisor_weighted_pixel_sum)
            # losses.append(torch.sum(loss_all) / torch.sum(mask.float()))
        return losses
#     def forward(self, inputs_scales, targets_scales):#(包含深监督的输出)
        
#         loss = 0
#         for input in inputs_scales:
#             targets_m_1 = targets_scales.clone() #深拷贝
#             targets_m = targets_m_1-1 #减去

#             loss_all = self.ce_loss(input, targets_m.long())

#             number_of_pixels_per_class = torch.bincount(targets_scales.flatten().type(self.dtype),
#                            minlength=self.num_classes)
#             divisor_weighted_pixel_sum = torch.sum(number_of_pixels_per_class[1:] * self.weight)   # without void

#             losses = loss_all / divisor_weighted_pixel_sum
#             loss += losses.item()

#         return loss

class CrossEntropyLoss2dForValidData:
    def __init__(self, device, weight, weighted_pixel_sum):
        super(CrossEntropyLoss2dForValidData, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            torch.from_numpy(np.array(weight)).float(),
            reduction='sum',
            ignore_index=-1
        )
        self.ce_loss.to(device)
        self.weighted_pixel_sum = weighted_pixel_sum
        self.total_loss = 0

    def add_loss_of_batch(self, inputs,targets):
        targets_m = targets.clone()
        targets_m -= 1
        loss = self.ce_loss(inputs, targets_m.long())
        self.total_loss += loss

    def compute_whole_loss(self):
        return self.total_loss.cpu().numpy().item() / self.weighted_pixel_sum.item()

    def reset_loss(self):
        self.total_loss = 0


class CrossEntropyLoss2dForValidDataUnweighted:
    def __init__(self, device):
        super(CrossEntropyLoss2dForValidDataUnweighted, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            weight=None,
            reduction='sum',
            ignore_index=-1
        )
        self.ce_loss.to(device)
        self.nr_pixels = 0
        self.total_loss = 0

    def add_loss_of_batch(self, inputs, targets):
        targets_m = targets.clone()
        targets_m -= 1
        loss = self.ce_loss(inputs, targets_m.long())
        self.total_loss += loss
        self.nr_pixels += torch.sum(targets_m >= 0)  # only non void pixels

    def compute_whole_loss(self):
        return self.total_loss.cpu().numpy().item() / self.nr_pixels.cpu().numpy().item()

    def reset_loss(self):
        self.total_loss = 0
        self.nr_pixels = 0

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


def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()

class DetailAggregateLoss(nn.Module):
    def __init__(self, inc, outc):
        super(DetailAggregateLoss, self).__init__()
        self.sobel_x1, self.sobel_y1 = get_sobel(inc, outc)

    def forward(self, boundary_logits, gtmasks):  # (1,1,480,640)/(1,480,640)
        gtmasks = gtmasks.unsqueeze(1)
        label_extract = run_sobel(self.sobel_x1, self.sobel_y1, gtmasks)
        label_extract[label_extract > 0.05] = 1
        label_extract[label_extract <= 0.05] = 0

        edge_loss = 0
        for boundary in boundary_logits:
            boundary_logits = torch.sigmoid(boundary)
            dice_loss = dice_loss_func(boundary_logits,label_extract)
            edge_loss += dice_loss.item()

        return edge_loss

def print_log(epoch, local_count, count_inter, dataset_size, loss, time_inter,
              learning_rates):
    print_string = 'Train Epoch: {:>3} [{:>4}/{:>4} ({: 5.1f}%)]'.format(
        epoch, local_count, dataset_size,
        100. * local_count / dataset_size)
    for i, lr in enumerate(learning_rates):
        print(lr)
        print_string += '   lr_{}: {:>6}'.format(i, round(lr, 10))
    print_string += '   Loss: {:0.6f}'.format(loss.item())
    print_string += '  [{:0.2f}s every {:>4} data]'.format(time_inter,
                                                          count_inter)
    print(print_string, flush=True)

def save_ckpt(ckpt_dir, model, optimizer, epoch):
    state = {
        'state_dict': model.state_dict(),
    }
    ckpt_model_filename = "ckpt_epoch_{}.pth".format(epoch)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print('{:>2} has been successfully saved'.format(path))


def save_ckpt_every_epoch(ckpt_dir, model, optimizer, epoch, best_miou,
                          best_miou_epoch):
    state = {
        'state_dict': model.state_dict()
    }
    ckpt_model_filename = "ckpt_latest.pth".format(epoch)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print('{:>2} has been successfully saved'.format(path))


def load_ckpt(model, optimizer, model_file, device):
    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        if device.type == 'cuda':
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(model_file,
                                    map_location=lambda storage, loc: storage)

        model.load_state_dict(checkpoint['state_dict'])

        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_file, checkpoint['epoch']))
        epoch = checkpoint['epoch']
        if 'best_miou' in checkpoint:
            best_miou = checkpoint['best_miou']
            print('Best mIoU:', best_miou)
        else:
            best_miou = 0

        if 'best_miou_epoch' in checkpoint:
            best_miou_epoch = checkpoint['best_miou_epoch']
            print('Best mIoU epoch:', best_miou_epoch)
        else:
            best_miou_epoch = 0
        return epoch, best_miou, best_miou_epoch
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        sys.exit(1)


def get_best_checkpoint(ckpt_dir, key='mIoU_test'):
    ckpt_path = None
    log_file = os.path.join(ckpt_dir, 'logs.csv')
    if os.path.exists(log_file):
        data = pd.read_csv(log_file)
        idx = data[key].idxmax()
        miou = data[key][idx]
        epoch = data.epoch[idx]
        ckpt_path = os.path.join(ckpt_dir, f'ckpt_epoch_{epoch}.pth')
    assert ckpt_path is not None, f'No trainings found at {ckpt_dir}'
    assert os.path.exists(ckpt_path), \
        f'There is no weights file named {ckpt_path}'
    print(f'Best mIoU: {100*miou:0.2f} at epoch: {epoch}')
    return ckpt_path

def WeightedMultiLabelSigmoidLoss(model_output, target):
    """
    model_output: BS X NUM_CLASSES X H X W
    target: BS X H X W X NUM_CLASSES
    true target:b c h w
    """
    # Calculate weight. (edge pixel and non-edge pixel)
    weight_sum = target.sum(dim=1).sum(dim=1).sum(dim=1).float().data  # BS
    edge_weight = weight_sum.data / float(target.size()[2] * target.size()[3])
    non_edge_weight = (target.size()[2] * target.size()[3] - weight_sum.data) / float(target.size()[2] * target.size()[3])
    one_sigmoid_out = torch.sigmoid(model_output)
    zero_sigmoid_out = 1 - one_sigmoid_out
    # target = target.transpose(1, 3).transpose(2, 3).float()  # BS X NUM_CLASSES X H X W
    loss = -non_edge_weight.unsqueeze(1).unsqueeze(2).unsqueeze(3) * target * torch.log(one_sigmoid_out.clamp(min=1e-10)) - \
           edge_weight.unsqueeze(1).unsqueeze(2).unsqueeze(3) * (1 - target) * torch.log(zero_sigmoid_out.clamp(min=1e-10))

    return loss.mean(dim=0).sum()


def JointEdgeSegLoss(input, target):
    log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1).cuda()
    target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1).cuda()
    target_trans = target_t.clone()

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)
    ignore_index = (target_t > 1)

    target_trans[pos_index] = 1
    target_trans[neg_index] = 0

    pos_index = pos_index.data.cpu().numpy().astype(bool)
    neg_index = neg_index.data.cpu().numpy().astype(bool)
    ignore_index = ignore_index.data.cpu().numpy().astype(bool)

    weight = torch.Tensor(log_p.size()).fill_(0)
    weight = weight.numpy()
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    weight[ignore_index] = 0

    weight = torch.from_numpy(weight)
    weight = weight.cuda()
    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight,reduction='mean')
    return loss
# model_output = torch.rand(1,40,5,5)
# target = torch.randint(2,size=(1,5,5,40))
# w = WeightedMultiLabelSigmoidLoss(model_output,target)
# print(w)
def cross_entropy_loss_RCF(predict, labelf):
    prediction = predict * labelf
    label = labelf.long()
    mask = labelf.clone()
    num_positive = torch.sum(label == 1).float()
    num_negative = torch.sum(label == 0).float()
    prediction = torch.sigmoid(prediction)

    mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[label == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[label == 2] = 0
    cost = F.binary_cross_entropy(
        prediction, labelf, weight=mask, reduction='sum')

    return cost
