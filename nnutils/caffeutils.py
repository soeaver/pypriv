import inspect, os, sys
import datetime
import cv2
import numpy as np

this_file = inspect.getfile(inspect.currentframe())
file_pth = os.path.abspath(os.path.dirname(this_file))
sys.path.append(file_pth + '/../')  # path of pypriv
import variable as V
import transforms as T
import basicmath as M

sys.path.append(file_pth + '/../../')  # path of py-RFCN-priv/lib
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.config import cfg


class CaffeInference(object):
    def __init__(self, net):
        self.net = net

    def inference(self, batch_ims, output_layer='prob'):
        input_ims = batch_ims.transpose(0, 3, 1, 2)
        self.net.blobs['data'].reshape(*input_ims.shape)
        self.net.blobs['data'].data[...] = input_ims
        self.net.forward()

        return self.net.blobs[output_layer].data[...]


class Classifier(CaffeInference):
    def __init__(self, net, class_num=1000, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0),
                 base_size=256, crop_size=224, crop_type='center', prob_layer='prob', synset=None):
        CaffeInference.__init__(self, net)
        self.class_num = class_num
        self.mean = mean
        self.std = std
        self.base_size = base_size
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.prob_layer = prob_layer
        self.synset = synset

    def cls_im(self, im):
        im = im.astype(np.float32, copy=True)
        normalized_im = T.normalize(im, mean=self.mean, std=self.std)
        scale_im = T.scale(normalized_im, short_size=self.base_size)
        crop_ims = []
        if self.crop_type == 'center' or self.crop_type == 'single':  # for single crop
            crop_ims.append(T.center_crop(scale_im, crop_size=self.crop_size))
        elif self.crop_type == 'mirror' or self.crop_type == 'multi':  # for 10 crops
            crop_ims.extend(T.mirror_crop(scale_im, crop_size=self.crop_size))
        else:
            crop_ims.append(scale_im)

        scores = self.inference(np.asarray(crop_ims, dtype=np.float32), output_layer=self.prob_layer)

        return np.sum(scores, axis=0)

    def cls_batch(self, batch_ims):
        input_ims = []
        for im in batch_ims:
            im = im.astype(np.float32, copy=True)
            normalized_im = T.normalize(im, mean=self.mean, std=self.std)
            scale_im = T.scale(normalized_im, short_size=self.base_size)
            input_ims.append(T.center_crop(scale_im, crop_size=self.crop_size))

        scores = self.inference(np.asarray(input_ims, dtype=np.float32), output_layer=self.prob_layer)

        return scores


class Identity(Classifier):
    def __init__(self, net, class_num=1000, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0),
                 base_size=256, crop_size=224, crop_type='center', prob_layer='prob', gallery=None):
        Classifier.__init__(self, net, class_num, mean, std, base_size, crop_size, crop_type, prob_layer)
        self.gallery = gallery

    def id_feature(self, im):
        return self.cls_im(im)

    def one2one_identity(self, im1, im2):
        normalized_im1 = T.normalize(im1, mean=self.mean, std=self.std)
        scale_im1 = T.scale(normalized_im1, short_size=self.base_size)
        input_im1 = T.center_crop(scale_im1, crop_size=self.crop_size)

        normalized_im2 = T.normalize(im2, mean=self.mean, std=self.std)
        scale_im2 = T.scale(normalized_im2, short_size=self.base_size)
        input_im2 = T.center_crop(scale_im2, crop_size=self.crop_size)

        batch = np.asarray([input_im1, input_im2], dtype=np.float32)
        scores = self.inference(batch, output_layer=self.prob_layer)

        return M.cosine_similarity(scores[0], scores[1])

    def one2n_identity(self, query, gallery=None):
        if gallery is None:
            gallery = self.gallery
        else:
            gallery = gallery
        sims = []

        for i in xrange(len(gallery)):
            sims.append(M.cosine_similarity(query, gallery[i]))

        return np.asarray(sims)


class Segmentor(CaffeInference):
    def __init__(self, net, class_num=21, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0),
                 scales=(256, 384, 512, 640, 768, 1024), crop_size=512, image_flip=False, crf=None):
        CaffeInference.__init__(self, net)
        self.class_num = class_num
        self.mean = mean
        self.std = std
        self.scales = scales
        self.crop_size = crop_size
        self.image_flip = image_flip
        self.crf = crf

        self.crf_factor = 1.0
        self.prob_layer = 'prob'
        self.stride_rate = 2 / 3.0

    def eval_im(self, im):
        im = im.astype(np.float32, copy=True)
        h, w = im.shape[:2]

        normalized_im = T.normalize(im, mean=self.mean, std=self.std)
        scale_ims, scale_ratios = T.multi_scale_by_max(normalized_im, scales=self.scales, image_flip=self.image_flip)

        score_map = np.zeros((h, w, self.class_num), dtype=np.float32)
        for _im, _ratio in zip(scale_ims, scale_ratios):
            if _ratio > 0:
                score_map += cv2.resize(self.scale_process(_im), (w, h))
            else:
                score_map += cv2.resize(self.scale_process(_im), (w, h))[:, ::-1]
        score_map /= len(self.scales)

        if self.crf:
            tmp_data = np.asarray([im.transpose(2, 0, 1)], dtype=np.float32)
            tmp_score = np.asarray([score_map.transpose(2, 0, 1)], dtype=np.float32)
            self.crf.blobs['data'].reshape(*tmp_data.shape)
            self.crf.blobs['data'].data[...] = tmp_data
            self.crf.blobs['data_dim'].data[...] = [[[h, w]]]
            self.crf.blobs['score'].reshape(*tmp_score.shape)
            self.crf.blobs['score'].data[...] = tmp_score * self.crf_factor
            self.crf.forward()
            score_map = self.crf.blobs[self.prob_layer].data[0].transpose(1, 2, 0)

        return score_map.argmax(2)

    def seg_im(self, im):
        """ Ignore self.scales; """
        im = im.astype(np.float32, copy=True)
        h, w = im.shape[:2]
        normalized_im = T.normalize(im, mean=self.mean, std=self.std)
        scale_im, scale_ratio = T.scale_by_max(normalized_im, long_size=self.crop_size)
        input_im = T.padding_im(scale_im, target_size=(self.crop_size, self.crop_size),
                                borderType=cv2.BORDER_CONSTANT)
        output = self.inference(np.asarray([input_im], dtype=np.float32))
        score = output[0].transpose(1, 2, 0)
        score_map = cv2.resize(score, None, None, fx=1. / scale_ratio, fy=1. / scale_ratio)[:h, :w, :]

        return score_map.argmax(2)

    def scale_process(self, scale_im):
        h, w = scale_im.shape[:2]
        im_size_min = min(h, w)
        im_size_max = max(h, w)

        if im_size_max <= self.crop_size:
            input_im = T.padding_im(scale_im, target_size=(self.crop_size, self.crop_size),
                                    borderType=cv2.BORDER_CONSTANT)
            output = self.inference(np.asarray([input_im], dtype=np.float32))
            score = output[0].transpose(1, 2, 0)[:h, :w, :]
        else:
            stride = np.ceil(self.crop_size * self.stride_rate)
            pad_im = scale_im
            if im_size_min < self.crop_size:
                pad_im = T.padding_im(scale_im, target_size=(self.crop_size, self.crop_size),
                                      borderType=cv2.BORDER_CONSTANT)

            ph, pw = pad_im.shape[:2]
            h_grid = int(np.ceil(float(ph - self.crop_size) / stride)) + 1
            w_grid = int(np.ceil(float(pw - self.crop_size) / stride)) + 1
            data_scale = np.zeros((ph, pw, self.class_num), dtype=np.float32)
            count_scale = np.zeros((ph, pw, self.class_num), dtype=np.float32)
            for grid_yidx in xrange(0, h_grid):
                for grid_xidx in xrange(0, w_grid):
                    s_x = int(grid_xidx * stride)
                    s_y = int(grid_yidx * stride)
                    e_x = min(s_x + self.crop_size, pw)
                    e_y = min(s_y + self.crop_size, ph)
                    s_x = int(e_x - self.crop_size)
                    s_y = int(e_y - self.crop_size)
                    sub_im = pad_im[s_y:e_y, s_x:e_x, :]
                    count_scale[s_y:e_y, s_x:e_x, :] += 1.0
                    input_im = T.padding_im(sub_im, target_size=(self.crop_size, self.crop_size),
                                            borderType=cv2.BORDER_CONSTANT)
                    output = self.inference(np.asarray([input_im], dtype=np.float32))
                    data_scale[s_y:e_y, s_x:e_x, :] += output[0].transpose(1, 2, 0)
            score = data_scale / count_scale
            score = score[:h, :w, :]

        return score


class Detector(object):
    def __init__(self, net, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0),
                 scales=(600,), max_sizes=(1000,), preN=6000, postN=300, conf_thresh=0.7,
                 color_map=V.COLORMAP81, class_map=V.PERSON_CLASS):
        self.net = net
        self.mean = mean
        self.std = std
        self.scales = scales
        self.max_sizes = max_sizes
        self.image_flip = False
        self.nms_thresh = 0.3
        self.box_vote = False
        self.conf_thresh = conf_thresh
        self.color_map = color_map
        self.class_map = class_map
        cfg.TEST.RPN_PRE_NMS_TOP_N = preN
        cfg.TEST.RPN_POST_NMS_TOP_N = postN
        cfg.USE_GPU_NMS = False

    def det_im(self, im):
        im = im.astype(np.float32, copy=True)
        normalized_im = T.normalize(im, mean=self.mean, std=self.std)
        scale_ims, scale_ratios = T.multi_scale(normalized_im, scales=self.scales, max_sizes=self.max_sizes,
                                                image_flip=self.image_flip)

        input_data = scale_ims[0].transpose(2, 0, 1)
        input_data = input_data.reshape((1,) + input_data.shape)
        self.net.blobs['data'].reshape(*input_data.shape)
        input_blob = {'data': input_data, 'rois': None}

        input_blob['im_info'] = np.array([[scale_ims[0].shape[0], scale_ims[0].shape[1], 1.0]], dtype=np.float32)
        self.net.blobs['im_info'].reshape(*input_blob['im_info'].shape)

        # do forward
        forward_kwargs = {'data': input_blob['data'].astype(np.float32, copy=False)}
        forward_kwargs['im_info'] = input_blob['im_info'].astype(np.float32, copy=False)
        output_blob = self.net.forward(**forward_kwargs)

        rois = self.net.blobs['rois'].data.copy()
        boxes = rois[:, 1:5] / scale_ratios[0]

        scores = output_blob['cls_prob']
        scores = scores.reshape(*scores.shape[:2])

        # Apply bounding-box regression deltas
        box_deltas = output_blob['bbox_pred']
        box_deltas = box_deltas.reshape(*box_deltas.shape[:2])
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, scale_ims[0].shape)

        objs = []
        for cls_ind, cls in enumerate(self.class_map[1:]):
            cls_ind += 1  # because we skipped background
            cls_boxes = pred_boxes[:, 4:8]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            inds = np.where(dets[:, 4] > self.conf_thresh)
            cls_dets = dets[inds]

            keep = nms(cls_dets, self.nms_thresh)
            dets_NMSed = cls_dets[keep, :]
            if self.box_vote:
                VOTEed = bbox_vote(dets_NMSed, cls_dets)
            else:
                VOTEed = dets_NMSed

            _obj = boxes_filter(VOTEed, cls, self.color_map[cls_ind], thresh=self.conf_thresh)
            objs.extend(_obj)

        return objs


def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)  # x1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)  # y1 >= 0
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)  # x2 < im_shape[1]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)  # y2 < im_shape[0]
    return boxes


def bbox_vote(dets_NMS, dets_all, thresh=0.5):
    dets_voted = np.zeros_like(dets_NMS)  # Empty matrix with the same shape and type

    _overlaps = bbox_overlaps(
        np.ascontiguousarray(dets_NMS[:, 0:4], dtype=np.float),
        np.ascontiguousarray(dets_all[:, 0:4], dtype=np.float))

    # for each survived box
    for i, det in enumerate(dets_NMS):
        dets_overlapped = dets_all[np.where(_overlaps[i, :] >= thresh)[0]]
        assert (len(dets_overlapped) > 0)

        boxes = dets_overlapped[:, 0:4]
        scores = dets_overlapped[:, 4]

        out_box = np.dot(scores, boxes)

        dets_voted[i][0:4] = out_box / sum(scores)  # Weighted bounding boxes
        dets_voted[i][4] = det[4]  # Keep the original score

    return dets_voted


def nms(dets, thresh):
    if dets.shape[0] == 0:
        return []
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def boxes_filter(dets, class_name, color, thresh=0.5, min_size=(2, 2)):
    """Draw detected bounding boxes."""
    _objs = []
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return _objs

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        if bbox[3] - bbox[1] <= min_size[0] or bbox[2] - bbox[0] <= min_size[1]:
            continue
        _objs.append(dict(bbox=bbox, score=score, class_name=class_name, color=color))

    return _objs

