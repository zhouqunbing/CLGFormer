# -*- coding: utf-8 -*-
from src.datasets.kitti import KITTI
import torchvision.transforms as standard_transforms
from torchvision import transforms
import src.transformers.joint_transforms as joint_transforms
import src.transformers.transforms as extended_transforms
from torch.utils.data import DataLoader


def setup_loaders(args):
    if args.dataset == 'kitti':
        Dataset = KITTI

    else:
        raise Exception('Dataset {} is not supported'.format(args.dataset))

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Geometric image transformations #几何变换
    train_joint_transform_list = [
        joint_transforms.RandomSizeAndCrop(args.crop_size,  # 720
                                           False,
                                           pre_size=None,  # None
                                           scale_min=args.aug_scale_min,  # 0.5
                                           scale_max=args.aug_scale_max,  # 2
                                           ignore_index=255),  # 255
        joint_transforms.Resize(args.crop_size),
        joint_transforms.RandomHorizontallyFlip()]
    train_joint_transform = joint_transforms.Compose(train_joint_transform_list)

    # Image appearance transformations #外观变换
    train_input_transform = []
    train_input_transform += [extended_transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
        saturation=0.25,
        hue=0.25)]
    train_input_transform += [extended_transforms.RandomGaussianBlur()]

    train_input_transform += [standard_transforms.ToTensor(), standard_transforms.Normalize(*mean_std)]
    train_input_transform = standard_transforms.Compose(train_input_transform)

    val_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    target_transform = extended_transforms.MaskToTensor()
    target_train_transform = extended_transforms.MaskToTensor()

    if args.dataset == 'kitti':
        # eval_size_h = 384
        # eval_size_w = 1280
        # val_joint_transform_list = [
        #         joint_transforms.ResizeHW(eval_size_h, eval_size_w)]

        train_set = Dataset(
            'semantic',
            'train',
            0,  # 0
            joint_transform_list=train_joint_transform_list,
            transform=train_input_transform,
            target_transform=target_train_transform,
            dump_images=False,  # False
            class_uniform_pct=args.class_uniform_pct,  # 0.5
            class_uniform_tile=args.class_uniform_tile,  # 1024
            test=False,  # False
            cv_split=args.cv,  # 3
            scf=False,  # False
            hardnm=0  # 0
        )
        val_set = Dataset(
            'semantic',
            'trainval',
            0,
            joint_transform_list=None,
            transform=val_input_transform,
            target_transform=target_transform,
            test=False,
            cv_split=args.cv,
            scf=None)

    else:
        raise Exception('Dataset {} is not supported'.format(args.dataset))

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              num_workers=args.workers,
                              shuffle=True,
                              drop_last=True,
                              )
    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size_valid,
                            num_workers=args.workers,
                            shuffle=False,
                            drop_last=False)

    return train_loader, val_loader,train_set

