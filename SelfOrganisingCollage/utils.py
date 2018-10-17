import time
import json
import torch
import math
from PIL import Image


convert_image = {
    0: lambda img: img,
    1: lambda img: img,
    2: lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
    3: lambda img: img.transpose(Image.ROTATE_180),
    4: lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),
    5: lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90),
    6: lambda img: img.transpose(Image.ROTATE_270),
    7: lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270),
    8: lambda img: img.transpose(Image.ROTATE_90),
}

def autorotate(img):
    try:
        img = convert_image[img._getexif().get(0x112, 1)](img)
    except AttributeError: # if it doesn't have exif data then there's nothing to be done
        pass

    return img


class TimeIt:
    def __init__(self, s):
        self.s = s

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, t, value, traceback):
        print('%s: %s' % (self.s, time.time() - self.t0))


def flatten(list_of_lists):
    return [item for l in list_of_lists for item in l]


def get_imagenet_class_labels():
    class_idx = json.load("imagenet_class_index.json")
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

    for idx in out[0].sort()[1][-10:]:
        print(idx2label[idx])


def simplify_shape(shape, target_area=16):
    aspect = shape[0] / shape[1]
    w = round(math.sqrt(target_area / aspect))
    h = round(w * aspect)
    return torch.Tensor([h,w]).int()


def ravel_index(idx, n_cols):
    return idx / n_cols, idx % n_cols
