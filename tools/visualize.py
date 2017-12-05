import inspect, os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

this_file = inspect.getfile(inspect.currentframe())
file_pth = os.path.abspath(os.path.dirname(this_file))

FONT0 = ImageFont.truetype(file_pth + '/../data/Arial.ttf', 20)
FONT1 = cv2.FONT_HERSHEY_SIMPLEX


def draw_bbox(im, objs, max_obj=100, draw_text=True):
    im = im.astype(np.float32, copy=True)
    for i in xrange(min(len(objs), max_obj)):
        bbox = objs[i]['bbox']
        cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), objs[i]['color'], 2)
        if draw_text:
            cv2.putText(im, '{:s} {:.3f}'.format(str(objs[i]['class_name']), objs[i]['score']),
                        (int(bbox[0] + 5), int(bbox[1] + 15)),
                        FONT1, 0.5, objs[i]['color'], thickness=1)

    return im


def draw_fancybbox(im, objs, max_obj=100, alpha=0.4):
    for i in xrange(min(len(objs), max_obj)):
        bbox = objs[i]['bbox']
        cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), objs[i]['color'], 2)
        vis = Image.fromarray(im)

        mask = Image.fromarray(im.copy())
        draw = ImageDraw.Draw(mask)
        draw.rectangle((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[1]) + 25), fill=tuple(objs[i]['color']))
        draw.text((int(bbox[0] + 5), int(bbox[1])),
                  '{:s} {:.3f}'.format(str(objs[i]['class_name']), objs[i]['score']),
                  fill=(255, 255, 255), font=FONT0)

        im = np.array(Image.blend(vis, mask, alpha))

    return im


def draw_mask(im, label, color_map, alpha=0.7):
    h, w = im.shape[:2]
    color = np.zeros((h, w, 3), dtype=np.uint8)

    category = np.unique(label)

    for c in list(category):
        color[np.where(label == c)] = color_map[c]

    mask = Image.blend(Image.fromarray(im), Image.fromarray(color), alpha)

    return np.array(mask)


def draw_pose_kpts(im, kpts, color_map, link_pair):
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
            if kpts[i][link[0]][2] * kpts[i][link[1]][2] == 0:
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


def draw_kpts(im, kpts):
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

