"""USIS10K loading for the MLP probes: same resize / normalize / pad as the mmdet test pipeline."""
import json
import math
import os.path as osp
import random

import cv2
import numpy as np
import torch


def load_split(data_root, split, n_images, seed=0):
    ann = json.load(open(osp.join(data_root, 'multi_class_annotations', f'multi_class_{split}_annotations.json')))
    by_img = {}
    for a in ann['annotations']:
        by_img.setdefault(a['image_id'], []).append(a)
    imgs = sorted(ann['images'], key=lambda d: d['id'])
    random.Random(seed).shuffle(imgs)
    return imgs[:n_images], by_img


def preprocess(path, norm, scale=(1333, 800)):
    """Returns the padded normalized (1,3,H,W) tensor, the resized RGB image and its (h, w)."""
    bgr = cv2.imread(path)
    h, w = bgr.shape[:2]
    s = min(scale[0] / max(h, w), scale[1] / min(h, w))  # mmcv imrescale, keep_ratio
    nh, nw = int(h * s + 0.5), int(w * s + 0.5)
    rgb = cv2.cvtColor(cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB).astype(np.float32)
    x = (rgb - np.array(norm['mean'], np.float32)) / np.array(norm['std'], np.float32)
    H, W = math.ceil(nh / 32) * 32, math.ceil(nw / 32) * 32
    xp = np.zeros((H, W, 3), np.float32)
    xp[:nh, :nw] = x
    return torch.from_numpy(xp).permute(2, 0, 1)[None], rgb, (nh, nw)


def valid_token_mask(hw, grid):
    """Tokens of the (Hg, Wg) 1/16 grid whose 16x16 patch lies fully inside the unpadded image."""
    Hg, Wg = grid
    m = np.zeros((Hg, Wg), bool)
    m[:hw[0] // 16, :hw[1] // 16] = True
    return m.reshape(-1)
