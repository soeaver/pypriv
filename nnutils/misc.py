import numpy as np


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


def boxes_filter(dets, bbox_id=1, class_name='None', color=(255, 255, 255), scale=1.0, thresh=0.5, min_size=(2, 2)):
    """Draw detected bounding boxes."""
    _objs = []
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return _objs

    for i in inds:
        bbox = dets[i, :4] / scale
        bbox_confidence = dets[i, -1]
        if bbox[3] - bbox[1] <= min_size[0] or bbox[2] - bbox[0] <= min_size[1]:
            continue
        attribute = dict(class_name=class_name, color=color)
        _objs.append(dict(bbox=bbox, bbox_id=bbox_id, bbox_confidence=bbox_confidence, attribute=attribute))

    return _objs
  
    
def objs_sort_by_center(objs, target=0):
    """target=0, sort by x; target=1, sort by y;."""
    sorted = []
    centers = []
    for i in objs:
        if target == 0:
            centers.append((i['bbox'][0] + i['bbox'][2]) / 2.0)
        elif target == 1:
            centers.append((i['bbox'][1] + i['bbox'][3]) / 2.0)
    centers_idx = np.argsort(np.asarray(centers))

    for i in centers_idx:
        sorted.append(objs[i])
        
    return sorted
