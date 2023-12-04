# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import tensorflow as tf
from src.args import ArgumentParserRGBDSegmentation
from src.build_model import build_model
from src.confusion_matrix import ConfusionMatrixTensorflow
from src.prepare_data import prepare_data
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
tf.compat.v1.disable_eager_execution()

def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1 # hxx
    imLab += 1 # label 应该是不用加的
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)

def label_accuracy_score(label_trues, label_preds, n_class, returniu=False):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    if returniu:
        return acc, acc_cls, mean_iu, fwavacc, iu[freq > 0]
    else:
        return acc, acc_cls, mean_iu, fwavacc

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2).reshape(
        n_class, n_class)
    return hist

if __name__ == '__main__':
    # arguments
    parser = ArgumentParserRGBDSegmentation(
        description='Efficient RGBD Indoor Sematic Segmentation (Evaluation)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    parser.add_argument('--ckpt_path', type=str,
                        required=True,
                        help='Path to the checkpoint of the trained model.')
    args = parser.parse_args()

    # dataset
    args.pretrained_on_imagenet = False  # we are loading other weights anyway
    _, data_loader, *add_data_loader = prepare_data(args, with_input_orig=True)
    if args.valid_full_res:
        # cityscapes only -> use dataloader that returns full resolution images
        data_loader = add_data_loader[0]

    n_classes = data_loader.dataset.n_classes_without_void

    # model and checkpoint loading
    model, device = build_model(args, n_classes=n_classes)
    checkpoint = torch.load(args.ckpt_path,map_location=lambda storage, loc: storage)
    # new_checkpoint = {f"module.{key}": val for key, val in checkpoint['state_dict'].items()}
    model.load_state_dict(checkpoint['state_dict'])
    # model.load_state_dict(new_checkpoint)
    print('Loaded checkpoint from {}'.format(args.ckpt_path))

    model.eval()
    model.to(device)

    n_samples = 0

    confusion_matrices = dict()

    cameras = data_loader.dataset.cameras

    label_trues, label_preds = [], []
    intersection_metric,union_metric = [],[]

    for camera in cameras:
        confusion_matrices[camera] = dict()
        confusion_matrices[camera] = ConfusionMatrixTensorflow(n_classes)
        n_samples_total = len(data_loader.dataset)
        with data_loader.dataset.filter_camera(camera):

            for i, sample in enumerate(data_loader):
                n_samples += sample['image'].shape[0]
                print(f'\r{n_samples}/{n_samples_total}', end='')

                image = sample['image'].to(device)
                depth = sample['depth'].to(device)
                label_orig = sample['label_orig'].to(device)

                _,h, w = label_orig.shape

                predicts = torch.zeros((3, 40, h, w), requires_grad=False).to(device)
                # eval_scales = [0.75, 0.75, 0.75, 1,   1,    1, 1.25, 1.25, 1.5, 1.5, 1.75, 1.75]
                # eval_scales = [0.75, 0.75, 0.75, 1,   1,    1, 1.25, 1.5, 1.5,1.5, 1.75, 1.75]       #51.99(51.990)
                # eval_scales = [0.75, 0.75, 0.75, 1,   1,    1, 1.25, 1.25, 1.5, 1.5, 1.75, 1.75] #51.99(51.985)
                #eval_scales = [0.75, 0.75, 1,    1,   1,    1.25, 1.5, 1.5, 1.75, 1.75]          #51.99(51.988)
                # eval_scales = [0.75, 0.75, 1,    1,   1,    1.25, 1.5, 1.75, 1.75]               #51.99
                # eval_scales = [0.75, 0.75, 0.75, 1, 1, 1, 1.25, 1.5, 1.5, 1.5, 1.75, 1.75]#  51.98
                # eval_scales = [0.75, 0.75, 0.75, 1, 1, 1, 1.25, 1.5, 1.5, 1.75, 1.75, 1.75]  #51.98
                # eval_scales = [0.75, 0.75, 1,    1,   1,    1.5, 1.75,1.75]                  #51.98
                # eval_scales = [0.75, 0.75, 1,    1,   1,    1.25, 1.25, 1.5, 1.75, 1.75]   #51.97
                # eval_scales = [0.75, 0.75, 0.75, 1, 1, 1, 1, 1.25, 1.25, 1.5, 1.5, 1.75, 1.75]#51.97
                # eval_scales = [0.75, 0.75, 1,    1,   1,    1.5, 1.5  1.75]        #51.97
                # eval_scales = [0.75, 0.75, 1,    1,   1,    1.5, 1.5, 1.75, 1.75]  #51.97
                # eval_scales = [0.75, 0.75, 1,    1,   1.5,  1.5, 1.75]             #51.96
                # eval_scales = [0.75, 0.75, 1,    1,   1,    1.5,  1.5, 1.5, 1.75]  #51.95
                # eval_scales = [0.75, 0.75, 1,    1, 1, 1, 1.25, 1.5, 1.75, 1.75]   #51.95
                # eval_scales = [0.75, 0.75, 1,    1,   1.25, 1.25, 1.5 ,1.5 1.75]   #51.94
                # eval_scales = [0.75, 0.75, 1.25, 1,   1,    1.5, 1.75]             #51.94
                # eval_scales = [0.75, 0.75, 1,    1,   1,    1.5, 1.75, 1.75, 1.75] #51.94
                # eval_scales = [0.75, 0.75, 1,    1,   1,    1,   1.5, 1.5,  1.75]  #51.91
                # eval_scales = [0.75, 1,    1.5,  1.75]                             #51.90
                # eval_scales = [0.75, 0.75, 1,    1,   1.5,  1.5, 1.75, 1.75]       #51.90
                # eval_scales = [0.75, 0.75, 1,    1,   1.5,  1.5, 1.75, 1.75]       #51.90
                # eval_scales = [0.75, 0.75, 0.75, 1, 1, 1, 1.5, 1.5, 1.75]          # 51.89
                # eval_scales = [0.75, 0.75, 1, 1, 1, 1.5, 1.5, 1.75, 1.75, 1.75]    # 51.89
                # eval_scales = [0.75, 0.75, 1,    1.5, 1.5   1.75]                  #51.87
                # eval_scales = [0.75, 0.75, 0.75, 1, 1, 1, 1.5, 1.5, 1.75]          #51.86
                # eval_scales = [0.75, 0.75, 1,    1,   1.5,  1.5]                   #51.82
                # eval_scales = [0.75, 0.75, 1, 1, 1, 1.5, 1.75]
                # eval_scales = [0.75, 1,    1,    1,   1.5,  1.5, 1.75,1.75, 1.75]  #51.59
                eval_scales = [1]
                with torch.no_grad():
                    for scale in eval_scales:
                    #     # newHW = (int((h * scale // 32) * 32), int((w * scale // 32) * 32))
                    #     # # newHW = [int(h * scale), int(w * scale)]
                    #     # new_image = F.interpolate(image, newHW, mode='bilinear', align_corners=True)
                    #     # new_depth = F.interpolate(depth, newHW, mode='bilinear', align_corners=True)
                    #     #
                    #     # out_o = model(new_image, new_depth)
                    #     # prob_o = F.softmax(out_o, 1)
                    #     #
                    #     # fout = model(torch.flip(new_image, dims=(3,)), torch.flip(new_depth, dims=(3,)))
                    #     # out_f = torch.flip(fout, dims=(3,))
                    #     # prob_o += F.softmax(out_f, 1)
                    #     #
                    #     # prob = torch.exp(prob_o)
                    #     #
                    #     # out = F.interpolate(prob_o, (h, w), mode='bilinear', align_corners=True)
                    #     # predicts += out
                        newHW = (int((h * scale // 32) * 32), int((w * scale // 32) * 32))
                        # newHW = [int(h * scale), int(w * scale)]
                        new_image = F.interpolate(image, newHW, mode='bilinear', align_corners=True)
                        new_depth = F.interpolate(depth, newHW, mode='bilinear', align_corners=True)

                        out_o = model(new_image, new_depth)
                        out_o = F.interpolate(out_o, (h, w), mode='bilinear', align_corners=True)
                        prob_o = F.softmax(out_o, 1)

                        fout = model(torch.flip(new_image, dims=(3,)), torch.flip(new_depth, dims=(3,)))
                        out_f = torch.flip(fout, dims=(3,))
                        out_f = F.interpolate(out_f, (h, w), mode='bilinear', align_corners=True)
                        prob_o += F.softmax(out_f, 1)
                        predicts += prob_o
                    # pred = model(image,depth)
                    # pred = F.interpolate(pred, (h, w),mode='bilinear',align_corners=False)
                    pred = torch.argmax(predicts, dim=1)
                    mask = label_orig > 0
                    label = torch.masked_select(label_orig, mask)
                    pred = torch.masked_select(pred, mask.to(device))
                    label -= 1

                    pred = pred.cpu()
                    label = label.cpu()

                    label = label.numpy()
                    pred = pred.numpy()

                    label_trues.append(label)
                    label_preds.append(pred)
                    intersection, union = intersectionAndUnion(pred, label, 40)

                    intersection_metric.append(intersection)
                    union_metric.append(union)


                    confusion_matrices[camera].update_conf_matrix(label, pred)

                print(f'\r{i + 1}/{len(data_loader)}', end='')

        iou = sum(intersection_metric) / (sum(union_metric) + 1e-10)
        metrics0 = label_accuracy_score(label_trues, label_preds, n_class=n_classes, returniu=True)
        metrics = np.array(metrics0[:4])
        metrics *= 100
        print('''\
                    Accuracy: {0}
                    Accuracy Class: {1}
                    Mean IU: {2}
                    FWAV Accuracy: {3}'''.format(*metrics))
        for i, _iou in enumerate(iou):
            print('class [{}], IoU: {}'.format(i, _iou))
        miou, _ = confusion_matrices[camera].compute_miou()
        print(f'\rCamera: {camera} mIoU: {100*miou:0.2f}')

    confusion_matrices['all'] = ConfusionMatrixTensorflow(n_classes)

    # sum confusion matrices of all cameras
    for camera in cameras:
        confusion_matrices['all'].overall_confusion_matrix += \
            confusion_matrices[camera].overall_confusion_matrix
    miou, _ = confusion_matrices['all'].compute_miou()
    print(f'All Cameras, mIoU: {100*miou:0.2f}')
