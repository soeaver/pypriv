import inspect, os
import numpy as np
import cv2
import math
from PIL import Image, ImageDraw, ImageFont

this_file = inspect.getfile(inspect.currentframe())
file_pth = os.path.abspath(os.path.dirname(this_file))

FONT10 = ImageFont.truetype(file_pth + '/../data/Arial.ttf', 10)
FONT15 = ImageFont.truetype(file_pth + '/../data/Arial.ttf', 15)
FONT20 = ImageFont.truetype(file_pth + '/../data/Arial.ttf', 20)
FONT30 = ImageFont.truetype(file_pth + '/../data/Arial.ttf', 30)
CVFONT0 = cv2.FONT_HERSHEY_SIMPLEX
CVFONT1 = cv2.FONT_HERSHEY_PLAIN


def draw_str(dst, (x, y), s):
    cv2.putText(dst, s, (x + 1, y + 1), CVFONT1, 1.0, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), CVFONT1, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)


def draw_bbox(im, objs, max_obj=100, draw_text=True):
    # im = im.astype(np.float32, copy=True)
    for i in xrange(min(len(objs), max_obj)):
        bbox = objs[i]['bbox']
        color = objs[i]['attribute']['color']
        class_name = objs[i]['attribute']['class_name']
        cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), objs[i]['color'], 2)
        if draw_text:
            cv2.putText(im, '{:s} {:.3f}'.format(str(class_name), objs[i]['bbox_confidence']),
                        (int(bbox[0] + 5), int(bbox[1] + 15)), CVFONT0, 0.5, color, thickness=1)

    return im


def draw_fancybbox(im, objs, max_obj=100, alpha=0.4, attri=False):
    vis = Image.fromarray(im)
    draw1 = ImageDraw.Draw(vis)
    mask = Image.fromarray(im.copy())
    draw2 = ImageDraw.Draw(mask)
    for i in xrange(min(len(objs), max_obj)):
        bbox = objs[i]['bbox']
        color = objs[i]['attribute']['color']
        class_name = objs[i]['attribute']['class_name']
        draw1.rectangle((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])), outline=tuple(color))
        draw2.rectangle((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[1]) + 24), fill=tuple(color))
        draw2.text((int(bbox[0] + 5), int(bbox[1]) + 2),
                   '{:s} {:.3f}'.format(str(class_name), objs[i]['bbox_confidence']), fill=(255, 255, 255), font=FONT20)

        if attri:
            attri_keys = objs[i]['attribute']['attri'].keys()
            y_shift = min(im.shape[0] - (int(bbox[1]) + 25 + 25 * len(attri_keys)), 0)
            left_top = (int(bbox[0]) - 110, int(bbox[1]) + 25 + y_shift)
            right_bottom = (int(bbox[0]) - 10, int(bbox[1]) + 25 + y_shift + 25 * len(attri_keys))
            draw1.rectangle((left_top[0], left_top[1], right_bottom[0], right_bottom[1]), fill=(32, 32, 32))
            for j in xrange(len(attri_keys)):
                draw1.text((left_top[0] + 5, left_top[1] + 2 + j * 25),
                          '{}: {}'.format(attri_keys[j], objs[i]['attribute']['attri'][attri_keys[j]]),
                          fill=(255, 255, 255), font=FONT15)

    return np.array(Image.blend(vis, mask, alpha))


def draw_fancybbox2(im, objs, max_obj=100, alpha=0.4, attri=False, line_factor=0.1):
    for i in xrange(min(len(objs), max_obj)):
        bbox = objs[i]['bbox']
        color = objs[i]['attribute']['color']
        class_name = objs[i]['attribute']['class_name']
        base_line = max(1, min(int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])))
        pts_left_top = np.array([[int(bbox[0]), int(bbox[1] + line_factor * base_line)],
                                 [int(bbox[0]), int(bbox[1])],
                                 [int(bbox[0] + line_factor * base_line), int(bbox[1])]], np.int32)
        pts_right_top = np.array([[int(bbox[2] - line_factor * base_line), int(bbox[1])],
                                  [int(bbox[2]), int(bbox[1])],
                                  [int(bbox[2]), int(bbox[1] + line_factor * base_line)]], np.int32)
        pts_right_bottom = np.array([[int(bbox[2]), int(bbox[3] - line_factor * base_line)],
                                     [int(bbox[2]), int(bbox[3])],
                                     [int(bbox[2] - line_factor * base_line), int(bbox[3])]], np.int32)
        pts_left_bottom = np.array([[int(bbox[0] + line_factor * base_line), int(bbox[3])],
                                    [int(bbox[0]), int(bbox[3])],
                                    [int(bbox[0]), int(bbox[3] - line_factor * base_line)]], np.int32)
        cv2.polylines(im, pts_left_top.reshape((1, -1, 1, 2)), False, (224, 224, 224),
                      thickness=min(3, int(base_line / 10.0)))
        cv2.polylines(im, pts_right_top.reshape((1, -1, 1, 2)), False, (224, 224, 224),
                      thickness=min(3, int(base_line / 10.0)))
        cv2.polylines(im, pts_right_bottom.reshape((1, -1, 1, 2)), False, (224, 224, 224),
                      thickness=min(3, int(base_line / 10.0)))
        cv2.polylines(im, pts_left_bottom.reshape((1, -1, 1, 2)), False, (224, 224, 224),
                      thickness=min(3, int(base_line / 10.0)))
        cv2.rectangle(im, (int(bbox[0] + 5), int(bbox[1] + 5)), (int(bbox[2] - 5), int(bbox[3] - 5)),
                      (224, 224, 224), 1)
        vis = Image.fromarray(im)

        mask = Image.fromarray(im.copy())
        draw = ImageDraw.Draw(mask)
        draw.rectangle((int(bbox[0] + 5), int(bbox[1] + 5),
                        int(bbox[2] - 5), int(bbox[1] + 30)),
                       fill=(32, 32, 32))
        draw.text((int(bbox[0] + 10), int(bbox[1]) + 7),
                  '{:s} {:.3f}'.format(str(objs[i]['class_name']), objs[i]['bbox_confidence']),
                  fill=(255, 255, 255), font=FONT20)
        if attri:
            attri_keys = objs[i]['attribute']['attri'].keys()
            y_shift = min(im.shape[0] - (int(bbox[1]) + 25 + 25 * len(attri_keys)), 0)
            left_top = (int(bbox[0]) - 110, int(bbox[1]) + 25 + y_shift)
            right_bottom = (int(bbox[0]) - 10, int(bbox[1]) + 25 + y_shift + 25 * len(attri_keys))
            draw.rectangle((left_top[0], left_top[1], right_bottom[0], right_bottom[1]), fill=(32, 32, 32))
            for j in xrange(len(attri_keys)):
                draw.text((left_top[0] + 5, left_top[1] + 2 + j * 25),
                          '{}: {}'.format(attri_keys[j], objs[i]['attribute']['attri'][attri_keys[j]]),
                          fill=(255, 255, 255), font=FONT15)

        im = np.array(Image.blend(vis, mask, alpha))

    return im


def draw_mask(im, label, color_map, alpha=0.7):
    h, w = im.shape[:2]
    color = im.astype(np.float32, copy=True)
    # color = np.zeros((h, w, 3), dtype=np.uint8)

    category = np.unique(label)

    for c in list(category):
        color[np.where(label == c)] = color_map[c]

    mask = Image.blend(Image.fromarray(im), Image.fromarray(color), alpha)

    return np.array(mask)


def draw_pose_kpts(im, kpts, color_map, link_pair, thresh=0):
    part_size = min(int(min(im.shape[:2]) / 100), 3)
    if kpts is None:
        return im
    p_num, k_num = kpts.shape[:2]  # n, 19, 3(x, y, score)
    for i in xrange(p_num):
        for j in xrange(k_num):
            if kpts[i][j][2] != 0:
                cv2.circle(im, (int(kpts[i][j][0]), int(kpts[i][j][1])), part_size, color_map[j], thickness=-1)

    for i in xrange(p_num):
        for j, link in enumerate(link_pair):
            if thresh > 0:
                k1 = 0 if kpts[i][link[0]][2] < thresh else 1
                k2 = 0 if kpts[i][link[1]][2] < thresh else 1
            else:
                k1 = kpts[i][link[0]][2]
                k2 = kpts[i][link[1]][2]
            if k1 * k2 == 0:
                continue
            cur_im = im.copy()
            mX = float(kpts[i][link[0]][0] * 0.5 + kpts[i][link[1]][0] * 0.5)
            mY = float(kpts[i][link[0]][1] * 0.5 + kpts[i][link[1]][1] * 0.5)
            length = ((kpts[i][link[0]][0] - kpts[i][link[1]][0]) ** 2 +
                      (kpts[i][link[0]][1] - kpts[i][link[1]][1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(kpts[i][link[0]][1] - kpts[i][link[1]][1],
                                            kpts[i][link[0]][0] - kpts[i][link[1]][0]))
            polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), part_size), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_im, polygon, color_map[j])
            im = cv2.addWeighted(im, 0.4, cur_im, 0.6, 0)

    return im


def draw_face68_kpts(im, kpts):
    links = [[0, 16], [17, 21], [22, 26], [27, 30], [31, 35], [36, 41], [42, 47], [48, 59], [60, 67]]
    if kpts is None:
        return im
    for _ in kpts:
        for x, y in _:
            cv2.circle(im, (int(x), int(y)), 2, (255, 255, 255), -1)

    for _ in xrange(len(kpts)):
        for i in links:
            for j in xrange(i[0], i[1]):
                cv2.line(im, (kpts[_][j][0], kpts[_][j][1]), (kpts[_][j + 1][0], kpts[_][j + 1][1]), (255, 255, 255), 1)
    return im

