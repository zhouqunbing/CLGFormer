"""
KITTI Dataset Loader
http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015
"""
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from PIL import Image
from torch.utils import data
import logging
import src.datasets.uniform as uniform
import src.datasets.cityscapes_labels as cityscapes_labels
import json

trainid_to_name = cityscapes_labels.trainId2name#(0:road)
id_to_trainid = cityscapes_labels.label2trainid#(0:255,1:255)
num_classes = 19
ignore_label = 255
root = '/root/ESANet/datasets/KITTI'
# aug_root = cfg.DATASET.KITTI_AUG_DIR
#调色板
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
           153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def get_train_val(cv_split, all_items):
    # 90/10 train/val split, three random splits for cross validation
    val_0 = [1,5,11,29,35,49,57,68,72,82,93,115,119,130,145,154,156,167,169,189,198]
    val_1 = [0,12,24,31,42,50,63,71,84,96,101,112,121,133,141,155,164,171,187,191,197]
    val_2 = [3,6,13,21,41,54,61,73,88,91,110,121,126,131,142,149,150,163,173,183,199]

    train_set = []
    val_set = []

    if cv_split == 0:
        for i in range(200):#0
            if i in val_0:#0
                val_set.append(all_items[i])#第0个(img,mask)添加到验证集。
            else:
                train_set.append(all_items[i])#
    elif cv_split == 1:
        for i in range(200):
            if i in val_1:
                val_set.append(all_items[i])
            else:
                train_set.append(all_items[i])
    elif cv_split == 2:
        for i in range(200):
            if i in val_2:
                val_set.append(all_items[i])
            else:
                train_set.append(all_items[i])
    else:
        logging.info('Unknown cv_split {}'.format(cv_split))
        sys.exit()

    return train_set, val_set

def make_dataset(quality, mode, maxSkip=0, cv_split=0, hardnm=0):
    items = []
    all_items = []
    aug_items = []

    assert quality == 'semantic'
    assert mode in ['train', 'val', 'trainval']
    # note that train and val are randomly determined, no official split

    img_dir_name = "training"
    img_path = os.path.join(root, img_dir_name, 'image_2')#/kitti/training/image_2
    mask_path = os.path.join(root, img_dir_name, 'semantic')#/kitti/training/semantic

    c_items = os.listdir(img_path)#/kitti/training/image_2下所有的图片
    c_items.sort()

    for it in c_items:
        item = (os.path.join(img_path, it), os.path.join(mask_path, it))#image下图片和标签下的图片组成一个元组
        all_items.append(item)
    logging.info('KITTI has a total of {} images'.format(len(all_items)))

    # split into train/val
    train_set, val_set = get_train_val(cv_split, all_items)

    if mode == 'train':
        items = train_set
    elif mode == 'val':
        items = val_set
    elif mode == 'trainval':
        items = train_set + val_set
    else:
        logging.info('Unknown mode {}'.format(mode))
        sys.exit()

    logging.info('KITTI-{}: {} images'.format(mode, len(items)))

    return items, aug_items

def make_test_dataset(quality, mode, maxSkip=0, cv_split=0):
    items = []
    assert quality == 'semantic'
    assert mode == 'test'

    img_dir_name = "testing"
    img_path = os.path.join(root, img_dir_name, 'image_2')

    c_items = os.listdir(img_path)
    c_items.sort()
    for it in c_items:
        item = (os.path.join(img_path, it), None)
        items.append(item)#对可迭代对象进行排序
    logging.info('KITTI has a total of {} test images'.format(len(items)))

    return items, []

class KITTI(data.Dataset):

    def __init__(self, quality, mode, maxSkip=0, joint_transform_list=None,
                 transform=None, target_transform=None, dump_images=False,
                 class_uniform_pct=0, class_uniform_tile=0, test=False,
                 cv_split=None, scf=None, hardnm=0):

        self.quality = quality#’semantic‘
        self.mode = mode#'train'/trainval
        self.maxSkip = maxSkip#0
        self.joint_transform_list = joint_transform_list#3个transformer
        self.transform = transform #train需要做的变换
        self.target_transform = target_transform#标签转换为tensor
        self.dump_images = dump_images#False
        self.class_uniform_pct = class_uniform_pct#0.5
        self.class_uniform_tile = class_uniform_tile#1024
        self.scf = scf#False
        self.hardnm = hardnm#0

        if cv_split:#交叉验证,0
            self.cv_split = cv_split
            assert cv_split < 10, \
                'expected cv_split {} to be < CV_SPLITS {}'.format(
                    cv_split, 4)
        else:
            self.cv_split = 0

        if self.mode == 'test':
            self.imgs, _ = make_test_dataset(quality, mode, self.maxSkip, cv_split=self.cv_split)
        else:#mode = 'train','validation'
            self.imgs, _ = make_dataset(quality, mode, self.maxSkip, cv_split=self.cv_split, hardnm=self.hardnm)#179对(img,mask)
        assert len(self.imgs), 'Found 0 images, please check the data set'

        # Centroids for GT data
        if self.class_uniform_pct > 0:#0.5
            if self.scf:
                json_fn = 'kitti_tile{}_cv{}_scf.json'.format(self.class_uniform_tile, self.cv_split)
            else:
                json_fn = 'kitti_tile{}_cv{}_{}_hardnm{}.json'.format(self.class_uniform_tile, self.cv_split, self.mode, self.hardnm)
            # if os.path.isfile(json_fn):#False
            #     with open(json_fn, 'r') as json_data:
            #         centroids = json.load(json_data)
            #     self.centroids = {int(idx): centroids[idx] for idx in centroids}
            # else:
            #     if self.scf:#False
            #         self.centroids = kitti_uniform.class_centroids_all(
            #             self.imgs,
            #             num_classes,
            #             id2trainid=id_to_trainid,
            #             tile_size=class_uniform_tile)
            #     else:
                self.centroids = uniform.class_centroids_all(
                        self.imgs,#179
                        num_classes,#19
                        id2trainid=id_to_trainid,
                        tile_size=class_uniform_tile)#1024
                with open(json_fn, 'w') as outfile:
                    json.dump(self.centroids, outfile, indent=4)

        self.build_epoch()

    def build_epoch(self, cut=False):
        if self.class_uniform_pct > 0:#0.5
            self.imgs_uniform = uniform.build_epoch(self.imgs,
                                                    self.centroids,
                                                    num_classes,
                                                    0.5)#0.5
        else:
            self.imgs_uniform = self.imgs

    def __getitem__(self, index):
        elem = self.imgs_uniform[index]
        centroid = None
        if len(elem) == 4:
            img_path, mask_path, centroid, class_id = elem
        else:
            img_path, mask_path = elem

        if self.mode == 'test':
            img, mask = Image.open(img_path).convert('RGB'), None
        else:
            img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # kitti scale correction factor
        if self.mode == 'train' or self.mode == 'trainval':

            img = img.resize((1248, 384), Image.BICUBIC)
            mask = mask.resize((1248, 384), Image.NEAREST)
        elif self.mode == 'val':
            width, height = 1242, 376
            img = img.resize((width, height), Image.BICUBIC)
            mask = mask.resize((width, height), Image.NEAREST)
        elif self.mode == 'test':
            img_keepsize = img.copy()
            width, height = 1280, 384
            img = img.resize((width, height), Image.BICUBIC)
        else:
            logging.info('Unknown mode {}'.format(mode))
            sys.exit()

        if self.mode != 'test':
            mask = np.array(mask)
            mask_copy = mask.copy()

            for k, v in id_to_trainid.items():#将标签的34类转换为19类
                mask_copy[mask == k] = v
            mask = Image.fromarray(mask_copy.astype(np.uint8))

        # Image Transformations
        if self.joint_transform_list is not None:
            for idx, xform in enumerate(self.joint_transform_list):
                if idx == 0 and centroid is not None:
                    # HACK
                    # We assume that the first transform is capable of taking
                    # in a centroid
                    img, mask = xform(img, mask, centroid)
                else:
                    img, mask = xform(img, mask)

        # Debug
        if self.dump_images and centroid is not None:#False
            outdir = './dump_imgs_{}'.format(self.mode)
            os.makedirs(outdir, exist_ok=True)
            dump_img_name = trainid_to_name[class_id] + '_' + img_name
            out_img_fn = os.path.join(outdir, dump_img_name + '.png')
            out_msk_fn = os.path.join(outdir, dump_img_name + '_mask.png')
            mask_img = colorize_mask(np.array(mask))
            img.save(out_img_fn)
            mask_img.save(out_msk_fn)

        if self.transform is not None:
            img = self.transform(img)
            if self.mode == 'test':
                img_keepsize = self.transform(img_keepsize)
                mask = img_keepsize
        if self.target_transform is not None:
            if self.mode != 'test':
                mask = self.target_transform(mask)

        return img, mask, img_name

    def __len__(self):
        return len(self.imgs_uniform)
