import numpy as np


def length_ab(pt_a, pt_b):
    ab = np.array([pt_a[0] - pt_b[0], pt_a[1] - pt_b[1]])
    len_ab = np.sqrt(ab.dot(ab))

    return len_ab


def cosine_similarity(vec1, vec2):
    len_vec1 = np.sqrt(vec1.dot(vec1))
    len_vec2 = np.sqrt(vec2.dot(vec2))

    if len_vec1 * len_vec2 == 0:
        return -1

    return vec1.dot(vec2) / (len_vec1 * len_vec2)


def angle_abc(pt_a, pt_b, pt_c):
    """ angle of aBc """
    ab = np.array([pt_a[0] - pt_b[0], pt_a[1] - pt_b[1]])
    cb = np.array([pt_c[0] - pt_b[0], pt_c[1] - pt_b[1]])
    sim = cosine_similarity(ab, cb)
    angle = np.arccos(sim)

    return angle * 360 / 2 / np.pi

