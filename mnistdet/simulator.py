import os
import numpy as np
import PIL
from typing import *
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from .config import DEFAULT_CONFIG

__all__ = ["DEFAULT_CONFIG", "load_configs", "generate_detection_image_with_annotation"]


def load_configs(path):
    return OmegaConf.load(path)


def generate_detection_image_with_annotation(
    src_data: Sequence[Tuple[Image.Image, int]],
    num_src: int,
    config: Optional[OmegaConf] = None,
) -> Tuple[Image.Image, List[List[int]], List[int]]:
    """
    params:
        src_data: torch Dataset class or List-like object [Image, Label]
        num_src:  number of images
        config:   simulator config

    output:
        image:  PIL Image
        boxes:  List[Box], Box is List[int] [x0, y0, x1, y1]
        labels: List[int], ground truth labels for each box
    """

    if config is None:
        config = DEFAULT_CONFIG

    canvas = config.canvas
    margin = config.margin
    stride = config.stride
    min_size = config.min_size
    max_size = config.max_size
    rotation = config.rotation
    keep_aspect = config.keep_aspect

    bgd = PIL.Image.fromarray(
        np.random.randint(100, 200, (5, 5, 3)).astype(np.uint8)
    ).resize(canvas, 3)
    bgd = PIL.ImageEnhance.Contrast(bgd).enhance(config.bgd_contrast)
    bgd = PIL.ImageEnhance.Brightness(bgd).enhance(config.bgd_bridgtness)

    xrange_ = range(0, canvas[0] - margin[0], stride[0])
    yrange_ = range(0, canvas[1] - margin[1], stride[1])
    idx_ = np.random.choice(len(src_data), num_src)
    dx_ = np.random.choice(xrange_, num_src, replace=False)
    dy_ = np.random.choice(yrange_, num_src, replace=False)
    w_ = np.random.choice(range(min_size, max_size + 1, 8), num_src)
    h_ = (
        np.random.choice(range(min_size, max_size + 1, 8), num_src)
        if not keep_aspect
        else w_
    )
    rot_ = np.random.choice(range(-rotation, rotation), num_src)

    lab_ = []
    box_ = []
    img_ = []
    for jdx, idx in enumerate(idx_):

        img, lab = src_data[idx]

        img = img.resize((w_[jdx], h_[jdx]), resample=3)
        img = img.rotate(rot_[jdx], resample=3, expand=True)
        img_.append(img)

        box = list(img.getbbox())

        dx, dy = dx_[jdx], dy_[jdx]

        box[0::2] += dx
        box[1::2] += dy
        box[0] = clip(box[0], 0, canvas[0])
        box[2] = clip(box[2], 0, canvas[0])
        box[1] = clip(box[1], 0, canvas[1])
        box[3] = clip(box[3], 0, canvas[1])

        if (box[2] - box[0]) * (box[3] - box[1]) > 0:
            lab_.append(lab)
            box_.append(box)
            color = np.random.randint(100, 200, (3,))
            fgd = PIL.ImageOps.colorize(img, black=color * 0.7, white=color).convert(
                "RGB"
            )
            fgd = PIL.ImageEnhance.Contrast(fgd).enhance(config.fgd_contrast)
            fgd = PIL.ImageEnhance.Brightness(fgd).enhance(config.fgd_bridgtness)
            msk = img.filter(PIL.ImageFilter.GaussianBlur(radius=config.blur_radius))
            bgd.paste(fgd, (dx, dy), msk)

    return bgd, box_, lab_


def clip(x, minval, maxval):
    if x < minval:
        return minval
    if x > maxval:
        return maxval
    return x
