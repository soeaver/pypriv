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
parser.add_argument('--gpu_id', type=int, help='gpu id for evaluation',
                    default=0)
parser.add_argument('--data_root', type=str, help='Path to imagenet validation',
                    default=ROOT_PTH + '/Database/ILSVRC2017')
parser.add_argument('--val_file', type=str, help='val_file',    # 2015 for senet and inception-v3
                    default=ROOT_PTH + '/Program/caffe-model/cls/ILSVRC2012_val_norm.txt')
parser.add_argument('--model_weights', type=str, help='model weights',
                    default=ROOT_PTH + '/Program/caffe-model/cls/resnet/resnet18-1x64d/resnet18-1x64d-merge.caffemodel')
parser.add_argument('--model_deploy', type=str, help='model_deploy', 
                    default=ROOT_PTH + '/Program/caffe-model/cls/resnet/resnet18-1x64d/deploy_resnet18-1x64d-merge.prototxt')

parser.add_argument('--ground_truth', type=bool, default=True, help='whether provide gt labels')
parser.add_argument('--prob_layer', type=str, default='prob', help='prob layer name')
parser.add_argument('--class_num', type=int, default=1000, help='predict classes number')
parser.add_argument('--skip_num', type=int, default=0, help='skip_num for evaluation')
parser.add_argument('--base_size', type=int, default=256, help='short size of images')
parser.add_argument('--crop_size', type=int, default=224, help='crop size of images')
parser.add_argument('--crop_type', type=str, default='center', choices=['center', 'multi'],
                    help='crop type of evaluation')
parser.add_argument('--batch_size', type=int, default=1, help='batch size of multi-crop test')
parser.add_argument('--top_k', type=int, nargs='+', default=[1, 5], help='top_k')
parser.add_argument('--save_score_vec', type=bool, default=False, help='whether save the score map')

args = parser.parse_args()

# ------------------ MEAN ---------------------
PIXEL_MEANS = np.array([103.52, 116.28, 123.675])  # for model convert from torch/pytorch
# PIXEL_MEANS = np.array([104.0, 117.0, 123.0])  # for resnet_v1, senet
# PIXEL_MEANS = np.array([104.0, 117.0, 124.0])  # for dpn
# PIXEL_MEANS = np.array([128.0, 128.0, 128.0])  # for inception_v3/v4/inception_resnet_v2
# PIXEL_MEANS = np.array([102.98, 115.947, 122.772])  # for resnet101(152,269)_v2
# ------------------ STD ---------------------
PIXEL_STDS = np.array([57.375, 57.12, 58.395])  # for model convert from torch/pytorch
# PIXEL_STDS = np.array([1.0, 1.0, 1.0])  # for resnet_v1, resnet101(152,269)_v2, senet
# PIXEL_STDS = np.array([59.88, 59.88, 59.88])  # for dpn
# PIXEL_STDS = np.array([128.0, 128.0, 128.0])  # for inception_v3/v4/inception_resnet_v2
# ---------------------------------------

if args.gpu_mode:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
else:
    caffe.set_mode_cpu()
net = caffe.Net(args.model_deploy, args.model_weights, caffe.TEST)

CLS = caffeutils.CaffeInference(net)

LOG_PTH = './log{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
SET_DICT = dict()
f = open(args.val_file, 'r')
img_order = 0
for _ in f:
    img_dict = dict()
    img_dict['path'] = os.path.join(args.data_root + _.strip().split(' ')[0])
    img_dict['evaluated'] = False
    img_dict['score_vec'] = []
    if args.ground_truth:
        img_dict['gt'] = int(_.strip().split(' ')[1])
    else:
        img_dict['gt'] = -1
    SET_DICT[img_order] = img_dict
    img_order += 1
f.close()


def shuffle_conv1_channel():
    conv1 = net.params['conv1'][0].data.copy()
    conv1[:, [0, 2], :, :] = conv1[:, [2, 0], :, :]
    net.params['conv1'][0].data[...] = conv1
    net.save(args.model_weights.replace('.caffemodel', '-bgr.caffemodel'))


def eval_batch():
    # shuffle_conv1_channel()
    eval_len = len(SET_DICT)
    # eval_len = 1000
    accuracy = np.zeros(len(args.top_k))
    start_time = datetime.datetime.now()

    for i in xrange(eval_len - args.skip_num):
        im = cv2.imread(SET_DICT[i + args.skip_num]['path'])
        if (PIXEL_MEANS == np.array([103.52, 116.28, 123.675])).all() and \
                (PIXEL_STDS == np.array([57.375, 57.12, 58.395])).all():
            scale_im = T.pil_scale(Image.fromarray(im), args.base_size)
            scale_im = np.asarray(scale_im)
        else:
            scale_im, _ = T.scale(im, short_size=args.base_size) 
        input_im = T.normalize(scale_im, mean=PIXEL_MEANS, std=PIXEL_STDS)
        crop_ims = []
        if args.crop_type == 'center':  # for single crop
            crop_ims.append(T.center_crop(input_im, crop_size=args.crop_size))
        elif args.crop_type == 'multi':  # for 10 crops
            crop_ims.extend(T.mirror_crop(input_im, crop_size=args.crop_size))
        else:
            crop_ims.append(input_im)

        score_vec = np.zeros(args.class_num, dtype=np.float32)
        iter_num = int(len(crop_ims) / args.batch_size)
        timer_pt1 = datetime.datetime.now()
        for j in xrange(iter_num):
            scores = CLS.inference(
                np.asarray(crop_ims, dtype=np.float32)[j * args.batch_size:(j + 1) * args.batch_size],
                output_layer=args.prob_layer
            )
            score_vec += np.sum(scores, axis=0)
        score_index = (-score_vec / len(crop_ims)).argsort()
        timer_pt2 = datetime.datetime.now()

        SET_DICT[i + args.skip_num]['evaluated'] = True
        SET_DICT[i + args.skip_num]['score_vec'] = score_vec / len(crop_ims)

        print 'Testing image: {}/{} {} {}/{} {}s' \
            .format(str(i + 1), str(eval_len - args.skip_num), str(SET_DICT[i + args.skip_num]['path'].split('/')[-1]),
                    str(score_index[0]), str(SET_DICT[i + args.skip_num]['gt']),
                    str((timer_pt2 - timer_pt1).microseconds / 1e6 + (timer_pt2 - timer_pt1).seconds)),

        for j in xrange(len(args.top_k)):
            if SET_DICT[i + args.skip_num]['gt'] in score_index[:args.top_k[j]]:
                accuracy[j] += 1
            tmp_acc = float(accuracy[j]) / float(i + 1)
            if args.top_k[j] == 1:
                print '\ttop_' + str(args.top_k[j]) + ':' + str(tmp_acc),
            else:
                print 'top_' + str(args.top_k[j]) + ':' + str(tmp_acc)
    end_time = datetime.datetime.now()

    w = open(LOG_PTH, 'w')
    s1 = 'Evaluation process ends at: {}. \nTime cost is: {}. '.format(str(end_time), str(end_time - start_time))
    s2 = '\nThe model is: {}. \nThe val file is: {}. \n{} images has been tested, crop_type is: {}, base_size is: {}, ' \
         'crop_size is: {}.'.format(args.model_weights, args.val_file, str(eval_len - args.skip_num),
                                    args.crop_type, str(args.base_size), str(args.crop_size))
    s3 = '\nThe PIXEL_MEANS is: ({}, {}, {}), PIXEL_STDS is : ({}, {}, {}).' \
        .format(str(PIXEL_MEANS[0]), str(PIXEL_MEANS[1]), str(PIXEL_MEANS[2]), str(PIXEL_STDS[0]), str(PIXEL_STDS[1]),
                str(PIXEL_STDS[2]))
    s4 = ''
    for i in xrange(len(args.top_k)):
        _acc = float(accuracy[i]) / float(eval_len - args.skip_num)
        s4 += '\nAccuracy of top_{} is: {}; correct num is {}.'.format(str(args.top_k[i]), str(_acc),
                                                                       str(int(accuracy[i])))
    print s1, s2, s3, s4
    w.write(s1 + s2 + s3 + s4)
    w.close()

    if args.save_score_vec:
        w = open(LOG_PTH.replace('.txt', 'scorevec.txt'), 'w')
        for i in xrange(eval_len - args.skip_num):
            w.write(SET_DICT[i + args.skip_num]['score_vec'])
    w.close()
    print('DONE!')


if __name__ == '__main__':
    eval_batch()

