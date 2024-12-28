"""
Copyright (c) 2024 TU Munich
Author: Nikita Araslanov <nikita.araslanov@tum.de>
License: Apache License 2.0
"""

import os
import math
import random
import glob
import torch
import torch.utils.data as data
import torchvision.transforms as tf_func

from PIL import Image, ImageFilter
import torch.nn.functional as F

class DavisVideo(data.Dataset):

    def __init__(self, data_root, split, crop_range, cutout_size, cutout_prob, cutout_num, input_size, gap, timeflip, temp_win):
        super().__init__()

        self.id = random.random()
        print("My ID = ", self.id)

        self.name = "DavisVideo"

        self.data_root = data_root
        self.split = split
        self.crop_range = crop_range
        self.cutout_size = cutout_size
        self.cutout_prob = cutout_prob
        self.cutout_num = cutout_num
        self.input_size = input_size
        self.gap = gap
        self.timeflip = timeflip
        self.temp_win = temp_win

        # train/val/test splits are pre-cut
        split_fn = os.path.join(self.data_root, f"{self.split}.txt")
        assert os.path.isfile(split_fn), f"File {split_fn} not found"

        def check_dir(path):
            full_path = os.path.join(self.data_root, path.lstrip('/'))
            return full_path

        def load_filenames(path):
            return sorted(glob.glob(path + "/*.jpeg") + glob.glob(path + "/*.jpg"))

        def load_masks(path):
            if path:
                return sorted(glob.glob(path + "/*.png"))
            return None

        self.videos = []

        with open(split_fn, "r") as lines:
            for n, line in enumerate(lines):
                paths = line.strip("\n").split(' ') + [None, None]
                image_dir, mask_dir = paths[:2]
                image_dir = check_dir(image_dir)

                if mask_dir:
                    mask_dir = check_dir(mask_dir)

                images = load_filenames(image_dir)
                masks = load_masks(mask_dir)
                self.videos.append({
                    "images": images,
                    "masks": masks,
                    "has_masks": mask_dir is not None,
                    "len": len(images)
                })

        total_num_frames = sum([len(v["images"]) for v in self.videos])
        print(f"Loaded {len(self.videos)} sequences | Total frames {total_num_frames}")

        self.tf_photo = GaussianBlur(p=0.5)
        self.tf_affine_crop = MaskRandScaleCrop(self.crop_range)

        cutouts = [cutout(self.cutout_size, self.cutout_prob) for _ in range(self.cutout_num)]
        self.tf_cutout = tf_func.Compose(cutouts)

        self.tf = tf_func.Compose([
            tf_func.Resize(self.input_size),
            tf_func.ToTensor(),
            tf_func.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.tf_norm = tf_func.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        index = index % len(self.videos)

        images = self.videos[index]["images"]
        frame_idx0 = random.randint(
            self.timeflip * self.gap * (self.temp_win - 1),
            len(images) - 1 - (self.temp_win - 1) * self.gap
        )
        frame_idx1 = frame_idx0 + random.randint(1, self.gap) * (1 - self.timeflip * random.choice([0, 2]))

        image0 = Image.open(images[frame_idx0]).convert('RGB')
        image0_ctr = tf_func.CenterCrop(min(image0.size[0], image0.size[1]))(image0)
        frame0 = self.tf(image0_ctr)

        image1 = Image.open(images[frame_idx1]).convert('RGB')
        image1_ctr = tf_func.CenterCrop(min(image1.size[0], image1.size[1]))(image1)
        frame1 = self.tf(image1_ctr)

        affine_params = self.tf_affine_crop(image0_ctr.size)
        affine_grid = F.affine_grid(
            affine_params[None, ...],
            (1, 1, self.input_size[0], self.input_size[1]),
            align_corners=False
        )

        image0_ctr = tf_func.ToTensor()(image0_ctr)
        frameIn = F.grid_sample(image0_ctr[None, ...], affine_grid, align_corners=False)
        frameIn = self.tf_norm(frameIn[0])
        frameInNs = self.tf_cutout(frameIn.clone())

        return torch.stack([frame0, frame1], 0), frameInNs, frameIn, affine_params






def cutout(mask_size, p):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        if random.random() > p:
            return image

        h, w = image.shape[-2:]

        # cutout_inside:
        cxmin, cxmax = mask_size_half, w + offset - mask_size_half
        cymin, cymax = mask_size_half, h + offset - mask_size_half

        cx = random.randint(cxmin, cxmax)
        cy = random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[:, ymin:ymax, xmin:xmax].zero_()
        return image

    return _cutout


class MaskRandScaleCrop(object):

    def __init__(self, scale_range):
        self.scale_from, self.scale_to = scale_range

    def get_params(self, h, w):
        # generating random crop
        # preserves aspect ratio
        new_scale = random.uniform(self.scale_from, self.scale_to)

        new_h = int(new_scale * h)
        new_w = int(new_scale * w)

        # generating
        if new_scale < 1.:
            assert w >= new_w, "{} vs. {} | {} / {}".format(w, new_w, h, new_h)
            i = random.randint(0, h - new_h)
            j = random.randint(0, w - new_w)
        else:
            assert w <= new_w, "{} vs. {} | {} / {}".format(w, new_w, h, new_h)
            i = random.randint(h - new_h, 0)
            j = random.randint(w - new_w, 0)

        return i, j, new_h, new_w, new_scale

    def get_affine_inv(self, affine, params, crop_size):

        aspect_ratio = crop_size[0] / crop_size[1]

        affine_inv = affine.clone()
        affine_inv[0, 1] = affine[1, 0] * aspect_ratio ** 2
        affine_inv[1, 0] = affine[0, 1] / aspect_ratio ** 2
        affine_inv[0, 2] = -1 * (affine_inv[0, 0] * affine[0, 2] + affine_inv[0, 1] * affine[1, 2])
        affine_inv[1, 2] = -1 * (affine_inv[1, 0] * affine[0, 2] + affine_inv[1, 1] * affine[1, 2])

        # scaling
        affine_inv /= torch.Tensor(params)[3].view(1, 1) ** 2

        return affine_inv

    def get_affine(self, params, crop_size):
        # construct affine operator
        affine = torch.zeros(2, 3)

        aspect_ratio = crop_size[0] / crop_size[1]  # float

        dy, dx, alpha, scale, flip = params

        # R inverse
        sin = math.sin(alpha * math.pi / 180.)
        cos = math.cos(alpha * math.pi / 180.)

        # inverse, note how flipping is incorporated
        affine[0, 0], affine[0, 1] = flip * cos, sin * aspect_ratio
        affine[1, 0], affine[1, 1] = -sin / aspect_ratio, cos

        # T inverse Rinv * t == R^T * t
        affine[0, 2] = -1. * (cos * dx + sin * dy)
        affine[1, 2] = -1. * (-sin * dx + cos * dy)

        # T
        affine[0, 2] /= crop_size[1] // 2  # integer
        affine[1, 2] /= crop_size[0] // 2  # integer

        # scaling
        affine *= scale

        affine = self.get_affine_inv(affine, params, crop_size)

        return affine

    def __call__(self, WH):

        affine = [0., 0., 0., 1., 1.]

        W, H = WH

        i2 = H / 2
        j2 = W / 2

        ii, jj, h, w, s = self.get_params(H, W)
        assert s < 1. and ii >= 0 and jj >= 0

        # displacement of the centre
        dy = ii + h / 2 - i2
        dx = jj + w / 2 - j2

        affine[0] = dy
        affine[1] = dx
        affine[3] = 1 / s  # scale

        # image_crop = tf_func.crop(image, ii, jj, h, w)
        # image_crop = image_crop.resize((W, H), Image.BILINEAR)

        return self.get_affine(affine, (H, W))


def get_color_distortion(s=0.5):
    # Credit: I-JEPA
    # s is the strength of color distortion.
    color_jitter = tf_func.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = tf_func.RandomApply([color_jitter], p=0.5)
    # rnd_gray = tf.RandomGrayscale(p=0.2)
    # color_distort = tf.Compose([
    #    rnd_color_jitter,
    #    rnd_gray])

    return rnd_color_jitter  # color_distort


class GaussianBlur(object):
    # Credit: I-JEPA

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

