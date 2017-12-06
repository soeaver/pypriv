import os, getpass

ROOT_PTH = os.path.abspath(os.path.join('/home', getpass.getuser()))

import cv2
import numpy as np

GT_ROOT = ROOT_PTH + '/Database/VOC_PASCAL/VOC2012_trainval/SegmentationClassAug'
PRE_ROOT = 'predict'
VAL_PTH = ROOT_PTH + '/Database/VOC_PASCAL/SBD/segmentation_set/voc2012val.txt'
CLASS_NUM = 21


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def compute_hist(val_list):
    hist = np.zeros((CLASS_NUM, CLASS_NUM))
    for idx in val_list:
        print idx
        label = cv2.imread('{}/{}.png'.format(GT_ROOT, idx), 0)
        gt = label.flatten()
        tmp = cv2.imread('{}/{}.png'.format(PRE_ROOT, idx), 0)

        if label.shape != tmp.shape:
            pre = cv2.resize(tmp, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_NN)
            pre = pre.flatten()
        else:
            pre = tmp.flatten()

        hist += fast_hist(gt, pre, CLASS_NUM)

    # return hist[1:, 1:]
    return hist


def mean_IoU(overall_h):
    iu = np.diag(overall_h) / (overall_h.sum(1) + overall_h.sum(0) - np.diag(overall_h))
    return np.nanmean(iu)


def per_class_acc(overall_h):
    acc = np.diag(overall_h) / overall_h.sum(1)
    return np.nanmean(acc)


def pixel_wise_acc(overall_h):
    return np.diag(overall_h).sum() / overall_h.sum()


if __name__ == '__main__':
    val_list = []

    f = open(VAL_PTH, 'r')
    for i in f:
        val_list.append(i.strip().split(' ')[-1].split('/')[-1])

    hist = compute_hist(val_list)

    print 'Mean IoU:', mean_IoU(hist)
    print 'Pixel Acc:', np.diag(hist).sum() / hist.sum()
    print 'Mean Acc:', per_class_acc(hist)

