# -*- coding: utf-8 -*-
import warnings
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch import nn

from src.models.model import CLGFormer
from src.models.resnet import ResNet


def build_model(args, n_classes):
    if not args.pretrained_on_imagenet or args.last_ckpt or \
            args.pretrained_scenenet != '':
        pretrained_on_imagenet = False
    else:
        pretrained_on_imagenet = True

    if args.modality == 'rgbd':
        if args.encoder_depth in [None, 'None']:
            args.encoder_depth = args.encoder

        model = CLGFormer(
            num_classes=n_classes,
            pretrained_on_imagenet=pretrained_on_imagenet,
            pretrained_dir=args.pretrained_dir,
            encoder_rgb=args.encoder,
            encoder_depth=args.encoder_depth,
            encoder_block=args.encoder_block,
            activation=args.activation,
            fuse_depth_in_rgb_encoder=args.fuse_depth_in_rgb_encoder,
        )
    device = torch.device("cuda")
    print('Device:', device)
    model.to(device)

    if args.he_init:
        module_list = []

        # first filter out the already pretrained encoder(s)
        for c in model.children():
            if pretrained_on_imagenet and isinstance(c, ResNet):
                # already initialized
                continue
            for m in c.modules():
                module_list.append(m)
        for i, m in enumerate(module_list):
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                if m.out_channels == n_classes or \
                        isinstance(module_list[i+1], nn.Sigmoid) or \
                        m.groups == m.in_channels:
                    continue
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        print('Applied He init.')

    if args.finetune is not None:
        checkpoint = torch.load(args.finetune)
        model.load_state_dict(checkpoint['state_dict'])
        print(f'Loaded weights for finetuning: {args.finetune}')

        # print('Freeze the encoder(s).')
        # for name, param in model.named_parameters():
        #     if 'encoder_rgb' in name or 'encoder_depth' in name or 'se_layer' in name:
        #         param.requires_grad = False

    return model, device
