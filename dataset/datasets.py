import os
import numpy as np
import random
import torch
import cv2
import json
from torch.utils import data
import torch.distributed as dist
from utils.transforms import get_affine_transform
import os.path as osp
from collections import defaultdict

class HelenDataSet(data.Dataset):
    def __init__(self, root, dataset, crop_size=[473, 473], scale_factor=0.25,
                 rotation_factor=30, ignore_label=255, transform=None):
        """
        :rtype:
        """
        self.root = root
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]
        self.crop_size = np.asarray(crop_size)
        self.ignore_label = ignore_label
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.flip_prob = 0.5
        self.flip_pairs = [[4, 5], [6, 7]]
        self.transform = transform
        self.dataset = dataset

        self.file_list_name = osp.join(root, dataset + '_list.txt')
        self.im_list = [line.split() for line in open(self.file_list_name).readlines()]
        self.number_samples = len(self.im_list)

    def __len__(self):
        return self.number_samples 

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)

        return center, scale

    def __getitem__(self, index):
        # Load training image
        image_name, label_name = self.im_list[index]
        im_name = os.path.splitext(image_name.replace('_image', ''))[0]

        im_path = os.path.join(self.root, image_name)
        edge_path = os.path.join(self.root, label_name.replace('labels', 'edges'))
        parsing_anno_path = os.path.join(self.root, label_name)
        assert os.path.exists(edge_path), print(edge_path)
        assert os.path.exists(im_path), print(im_path)
        assert os.path.exists(parsing_anno_path), print(parsing_anno_path)

        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        h, w, _ = im.shape
        parsing_anno = np.zeros((h, w), dtype=np.long)

        # Get center and scale
        center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0

        parsing_anno = cv2.imread(parsing_anno_path, cv2.IMREAD_GRAYSCALE)

        # Image clipping for augmentation
        if self.dataset == 'train':

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0

        trans = get_affine_transform(center, s, r, self.crop_size)
        input = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        edge = cv2.warpAffine(
            edge,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        edge[np.where(edge != 0)] = 1
        label_parsing = cv2.warpAffine(
            parsing_anno,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255))
        bi_label_parsing = np.copy(label_parsing)
        for mask_id in range(11):
            if mask_id in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
                bi_label_parsing[bi_label_parsing == mask_id] = 1
            else:
                bi_label_parsing[bi_label_parsing == mask_id] = 0

        if self.transform:
            input = self.transform(input)

        meta = {
            'name': im_name,
            'center': center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        if self.dataset in 'train':

            label_parsing = torch.from_numpy(label_parsing)
            bi_label_parsing = torch.from_numpy(bi_label_parsing)
            edge = torch.from_numpy(edge)

        return input, bi_label_parsing, label_parsing, edge, meta
