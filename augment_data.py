import numpy as np
import cv2
import random
from skimage.transform import resize


def zscore_norm(img):
    mean = img.mean()
    std = img.std()
    pixels = (img - mean) / std
    return pixels  # return float


def minmax_norm(img):
    minv = np.min(img)
    maxv = np.max(img)
    return (img - minv) / (maxv - minv)


def to_uint8(img):
    return (img * 255.).astype(np.uint8)


def rotate_image(img, angle=random.randint(-15, 15)):
    center = (int(img.shape[0] / 2), int(img.shape[1] / 2))
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rot_img = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))
    return rot_img


def flip_image(img):
    return np.flip(img, axis=1)


def stretch_image(img, lower_limit=0.8, upper_limit=1.2):
    stretch = random.uniform(lower_limit, upper_limit)
    height = int(img.shape[0] / np.sqrt(stretch))
    width = int(img.shape[1] * np.sqrt(stretch))
    stretched_img = resize(img, (width, height))

    l = min([width, height])  # new img size - assume to be square
    offy = int((stretched_img.shape[0] - l) / 2.)  # offsets to crop middle
    offx = int((stretched_img.shape[1] - l) / 2.)
    return resize(stretched_img[offy:offy+l, offx:offx+l], img.shape)


def augment_img(img, rand_chances, rs):
    actually_augmented = False
    augm_img = img
    for i, (aug_func, chance) in enumerate(rand_chances.items()):
        if rs[i] < chance:
            augm_img = aug_func(augm_img)
            actually_augmented = True
    return augm_img, actually_augmented
