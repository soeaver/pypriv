import inspect, os, sys, getpass

for idx, _ in enumerate(sys.path):
    if 'detectron' in _:
        del sys.path[idx]
        break

ROOT_PTH = os.path.abspath(os.path.join('/home', getpass.getuser()))
sys.path.append(ROOT_PTH + '/workspace/py-RFCN-priv/caffe-priv/python')
sys.path.append(ROOT_PTH + '/workspace/py-RFCN-priv/lib')
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
import variable as V

parser = argparse.ArgumentParser(description='Evaluat the Detection Model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--gpu_mode', type=bool, default=True, help='whether use gpu')
parser.add_argument('--gpu_id', type=int, default=4, help='gpi id for evaluation')
parser.add_argument('--data_root', type=str, default=ROOT_PTH + '/Database/Priv_personpart/Images/',
                    help='Path to imagenet validation path')
parser.add_argument('--val_file', type=str,
                    default=ROOT_PTH + '/Database/Priv_personpart/ImageSets/privpersonpart_val.txt',
                    help='val_file')
parser.add_argument('--image_ext', type=str, default='.jpg', help='suffix of the images')
parser.add_argument('--skip_num', type=int, default=0, help='skip_num for evaluation')
parser.add_argument('--model_weights', type=str,
                    default=ROOT_PTH + '/Program/caffe-model/det/rfcn/models/priv_personpart/resnet18/ms-lighthead-ohem/rfcn-lighthead_ppp_resnet18-priv-merge_ms7-roi512_iter_100000.caffemodel',
                    help='model weights')
parser.add_argument('--model_deploy', type=str,
                    default=ROOT_PTH + '/Program/caffe-model/det/rfcn/models/priv_personpart/resnet18/ms-lighthead-ohem/deploy_rfcn-lighthead_ppp_resnet18-priv-merge.prototxt',
                    help='model_deploy')
parser.add_argument('--save_root', type=str, default='predict', help='whether save the score map')


parser.add_argument('--class_num', type=int, default=4, help='predict classes number')
# parser.add_argument('--scales', type=int, nargs='+', default=[256, 384, 512, 640, 768, 1024], help='scales of image')
parser.add_argument('--scales', type=int, nargs='+', default=[600, ], help='scales of image')
parser.add_argument('--max_sizes', type=int, nargs='+', default=[1000, ], help='max_sizes of image')
parser.add_argument('--preN', type=int, default=6000, help='TEST.RPN_PRE_NMS_TOP_N')
parser.add_argument('--postN', type=int, default=500, help='TEST.RPN_POST_NMS_TOP_N')
parser.add_argument('--nms_thresh', type=float, default=0.5, help='nms thresh')
parser.add_argument('--conf_thresh', type=float, default=0.05, help='confidence thresh')

args = parser.parse_args()

# ------------------ MEAN ---------------------
# PIXEL_MEANS = np.array([0.0, 0.0, 0.0])  # for resnet10
# PIXEL_MEANS = np.array([104.0, 117.0, 123.0])  # for resnet_v1/resnet_custom/senet
PIXEL_MEANS = np.array([103.52, 116.28, 123.675])  # for resnext/wrn/densenet
# PIXEL_MEANS = np.array([128.0, 128.0, 128.0])  # for inception_v3/v4/inception_resnet_v2
# PIXEL_MEANS = np.array([102.98, 115.947, 122.772])  # for resnet101(152,269)_v2
# ------------------ STD ---------------------
# PIXEL_STDS = np.array([58.82, 58.82, 58.82])  # for senet
PIXEL_STDS = np.array([57.375, 57.12, 58.395])  # for resnext/wrn/densenet
# PIXEL_STDS = np.array([128.0, 128.0, 128.0])  # for inception_v3/v4/inception_resnet_v2
# PIXEL_STDS = np.array([1.0, 1.0, 1.0])  # for resnet101(152,269)_v2, resnet10, resnet_v1/resnet_custom
# ---------------------------------------

if args.gpu_mode:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
else:
    caffe.set_mode_cpu()
net = caffe.Net(args.model_deploy, args.model_weights, caffe.TEST)

DET = caffeutils.Detector(net, mean=PIXEL_MEANS, std=PIXEL_STDS, agnostic=True, scales=tuple(args.scales),
                          max_sizes=tuple(args.max_sizes), preN=args.preN, postN=args.postN, nms_thresh=args.nms_thresh,
                          conf_thresh=args.conf_thresh, color_map=V.COLORMAP81, class_map=V.PERSONPART_CLASS)


def eval_batch():
    eval_images = []
    f = open(args.val_file, 'r')
    for i in f:
        eval_images.append(i.strip())
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    eval_len = len(eval_images)
    all_boxes = [[[] for _ in xrange(eval_len - args.skip_num)] for _ in xrange(args.class_num)]

    start_time = datetime.datetime.now()
    for i in xrange(eval_len - args.skip_num):
        im = cv2.imread('{}{}{}'.format(args.data_root, eval_images[i + args.skip_num], args.image_ext))
        timer_pt1 = datetime.datetime.now()
        pre = DET.det_im(im)
        for _obj in pre:
            _idx = _obj['bbox_id']
            _bbox = _obj['bbox']
            all_boxes[_idx][i].append([_bbox[0], _bbox[1], _bbox[2], _bbox[3], _obj['bbox_confidence']])
        timer_pt2 = datetime.datetime.now()

        print 'Testing image: {}/{} {} {}s' \
            .format(str(i + 1), str(eval_len - args.skip_num), str(eval_images[i + args.skip_num]),
                    str((timer_pt2 - timer_pt1).microseconds / 1e6 + (timer_pt2 - timer_pt1).seconds))

    # det_file = './detections.pkl'
    # with open(det_file, 'wb') as f:
    #     cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    for cls_ind, cls in enumerate(V.PERSONPART_CLASS):
        if cls == '__background__':
            continue
        print 'Writing {} VOC format results file'.format(cls)
        filename = args.save_root + '/comp4' + '_det' + '_test_' + cls + '.txt'
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(eval_images[args.skip_num:]):
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                for k in xrange(len(dets)):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k][-1], dets[k][0] + 1, dets[k][1] + 1, dets[k][2] + 1, dets[k][3] + 1))
    end_time = datetime.datetime.now()
    print '\nEvaluation process ends at: {}. \nTime cost is: {}. '.format(str(end_time), str(end_time - start_time))


if __name__ == '__main__':
    eval_batch()
