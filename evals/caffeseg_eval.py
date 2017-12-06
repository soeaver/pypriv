import inspect, os, sys, getpass

ROOT_PTH = os.path.abspath(os.path.join('/home', getpass.getuser()))
sys.path.append(ROOT_PTH + '/workspace/py-RFCN-priv/caffe-priv/python')
this_file = inspect.getfile(inspect.currentframe())
file_pth = os.path.abspath(os.path.dirname(this_file))
sys.path.append(file_pth + '/../')  # path of pypriv

import numpy as np
import caffe
import cv2
import datetime
import argparse
import os
from nnutils import caffeutils
import transforms as T

parser = argparse.ArgumentParser(description='Evaluat the imagenet validation',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--gpu_mode', type=bool, default=True, help='whether use gpu')
parser.add_argument('--gpu_id', type=int, default=0, help='gpi id for evaluation')
parser.add_argument('--data_root', type=str, default=ROOT_PTH + '/Database/VOC_PASCAL/VOC2012_trainval/JPEGImages/',
                    help='Path to imagenet validation path')
parser.add_argument('--val_file', type=str,
                    default=ROOT_PTH + '/Database/VOC_PASCAL/SBD/segmentation_set/voc2012val.txt',
                    help='val_file')
parser.add_argument('--model_weights', type=str,
                    default=ROOT_PTH + '/Program/caffe-model/seg/pspnet/models/pascal_voc/se-resnet50/sbd_voc2012train/psp_se-resnet50_sbd_voc2012train_iter_20000.caffemodel',
                    help='model weights')
parser.add_argument('--model_deploy', type=str,
                    default=ROOT_PTH + '/Program/caffe-model/seg/pspnet/models/pascal_voc/se-resnet50/sbd_voc2012train/deploy_psp_se-resnet50-hik-merge.prototxt',
                    help='model_deploy')
parser.add_argument('--save_root', type=str, default='predict', help='whether save the score map')


parser.add_argument('--prob_layer', type=str, default='prob', help='prob layer name')
parser.add_argument('--class_num', type=int, default=21, help='predict classes number')
parser.add_argument('--skip_num', type=int, default=0, help='skip_num for evaluation')
# parser.add_argument('--scales', type=int, nargs='+', default=[256, 384, 512, 640, 768, 1024], help='scales of image')
parser.add_argument('--scales', type=int, nargs='+', default=[512, ], help='scales of image')
parser.add_argument('--crop_size', type=int, default=512, help='crop size of images')
parser.add_argument('--image_flip', type=bool, default=False, help='whether flip the image')

parser.add_argument('--crf', type=str, default=None, help='crf path')
parser.add_argument('--crf_factor', type=float, default=1.0, help='crf_factor')

args = parser.parse_args()

# ------------------ MEAN ---------------------
# PIXEL_MEANS = np.array([0.0, 0.0, 0.0])  # for resnet10
PIXEL_MEANS = np.array([104.0, 117.0, 123.0])  # for resnet_v1/resnet_custom/senet
# PIXEL_MEANS = np.array([103.52, 116.28, 123.675])  # for resnext/wrn/densenet
# PIXEL_MEANS = np.array([128.0, 128.0, 128.0])  # for inception_v3/v4/inception_resnet_v2
# PIXEL_MEANS = np.array([102.98, 115.947, 122.772])  # for resnet101(152,269)_v2
# ------------------ STD ---------------------
PIXEL_STDS = np.array([58.82, 58.82, 58.82])  # for senet
# PIXEL_STDS = np.array([57.375, 57.12, 58.395])  # for resnext/wrn/densenet
# PIXEL_STDS = np.array([128.0, 128.0, 128.0])  # for inception_v3/v4/inception_resnet_v2
# PIXEL_STDS = np.array([1.0, 1.0, 1.0])  # for resnet101(152,269)_v2, resnet10, resnet_v1/resnet_custom
# ---------------------------------------

if args.gpu_mode:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
else:
    caffe.set_mode_cpu()
net = caffe.Net(args.model_deploy, args.model_weights, caffe.TEST)

SEG = caffeutils.Segmentor(net, class_num=args.class_num, mean=PIXEL_MEANS, std=PIXEL_STDS, scales=tuple(args.scales),
                           crop_size=args.crop_size, image_flip=args.image_flip, crf=args.crf)


def eval_batch():
    eval_images = []
    f = open(args.val_file, 'r')
    for i in f:
        eval_images.append(i.strip())
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    eval_len = len(eval_images)
    start_time = datetime.datetime.now()
    for i in xrange(eval_len - args.skip_num):
        im = cv2.imread('{}{}.jpg'.format(args.data_root, eval_images[i + args.skip_num]))
        timer_pt1 = datetime.datetime.now()
        pre = SEG.eval_im(im)
        timer_pt2 = datetime.datetime.now()
        cv2.imwrite('{}/{}.png'.format(args.save_root, eval_images[i + args.skip_num]), pre)

        print 'Testing image: {}/{} {} {}s' \
            .format(str(i + 1), str(eval_len - args.skip_num), str(eval_images[i + args.skip_num]),
                    str((timer_pt2 - timer_pt1).microseconds / 1e6 + (timer_pt2 - timer_pt1).seconds))

    end_time = datetime.datetime.now()
    print '\nEvaluation process ends at: {}. \nTime cost is: {}. '.format(str(end_time), str(end_time - start_time))
    print '\n{} images has been tested. \nThe model is: {}'.format(str(eval_len), args.model_weights)


if __name__ == '__main__':
    eval_batch()

