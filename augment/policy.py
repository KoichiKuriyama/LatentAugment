from PIL import Image, ImageEnhance, ImageOps, ImageDraw
import numpy as np
import random
from torchvision.transforms import ToPILImage, ToTensor
# Use mixup
# https://github.com/DeepVoltaire/AutoAugment

# from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
def rotate_with_fill(img, img2, t, t2, magnitude):
    rot = img.convert("RGBA").rotate(magnitude)
    return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img1.mode), t

# https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/augmentations.py
def rotate_2(img, img2, t, t2, magnitude):  # [-30, 30]
    assert -30 <= magnitude <= 30
    if random.random() > 0.5:
        magnitude = -magnitude
    return img.rotate(magnitude), t

def shearX(img, img2, t, t2, magnitude):
    return img.transform(
        img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
        Image.BICUBIC, fillcolor=(128, 128, 128)), t

def shearY(img, img2, t, t2, magnitude):
    return  img.transform(
        img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
        Image.BICUBIC, fillcolor=(128, 128, 128)), t

def translateX(img, img2, t, t2, magnitude):
    return img.transform(
        img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
        fillcolor=(128, 128, 128)), t

def translateY(img, img2, t, t2, magnitude):
    return img.transform(
        img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
        fillcolor=(128, 128, 128)), t

def rotate(img, img2, t, t2, magnitude):
    return rotate_2(img, img2, t, t2, magnitude)

def color(img, img2, t, t2, magnitude):
    return ImageEnhance.Color(img).enhance(magnitude), t

def posterize(img, img2, t, t2, magnitude):
    return ImageOps.posterize(img, magnitude), t

def solarize(img, img2, t, t2, magnitude):
    return ImageOps.solarize(img, magnitude), t

def contrast(img, img2, t, t2, magnitude):
    return ImageEnhance.Contrast(img).enhance(magnitude), t

def sharpness(img, img2, t, t2, magnitude):
    return ImageEnhance.Sharpness(img).enhance(magnitude), t

def brightness(img, img2, t, t2, magnitude):
    return ImageEnhance.Brightness(img).enhance(magnitude), t

def autocontrast(img, img2, t, t2, magnitude):
    return ImageOps.autocontrast(img), t

def equalize(img, img2, t, t2, magnitude):
    return ImageOps.equalize(img), t

def invert(img, img2, t, t2, magnitude):
    return ImageOps.invert(img), t

def cutout(img, img2, t, t2, magnitude):
    return CutoutAbs(img, magnitude), t

def nothing(img, img2, t, t2, magnitude):
    return img, t

def samplepairing(img, img2, t, t2, magnitude):
    #return Image.blend(img, img2, 0.5)
    return ToPILImage()(0.5 * ToTensor()(img) + (1 - 0.5) * ToTensor()(img2)),t

def mixup(x1, x2, y1, y2, alpha=1.0):
    #https://www.inference.vc/mixup-data-dependent-data-augmentation/
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    mixed_x = ToPILImage()(lam * ToTensor()(x1) + (1 - lam) * ToTensor()(x2))
    mixed_y = lam * y1 + (1 - lam) * y2
    return mixed_x, mixed_y


class LatentPolicy(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        all_transforms = [
            "shearX",
            "shearY",
            "translateX",
            "translateY",
            "rotate",
            "color",
            "posterize",
            "solarize",
            "contrast",
            "sharpness",
            "brightness",
            "autocontrast",
            "equalize",
            "invert",
            "cutout",
            "mixup",
            ]

        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(-30, 30, 10),
            "color": np.linspace(0.1, 1.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.1, 1.9, 10),
            "sharpness": np.linspace(0.1, 1.9, 10),
            "brightness": np.linspace(0.1, 1.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10,
            "cutout": np.linspace(0, 20, 10),
            "mixup": [0] * 10,
        }

        self.all_transforms = all_transforms
        self.ranges = ranges

    def __call__(self, img, img2, img3, t, t2, t3, operation1, operation2):
        magnitude_idx1 = random.randint(0, 9)
        magnitude_idx2 = random.randint(0, 9)
        magnitude1 = self.ranges[self.all_transforms[operation1]][magnitude_idx1]
        magnitude2 = self.ranges[self.all_transforms[operation2]][magnitude_idx2]
        str1 = self.all_transforms[operation1]+"(img, img2, t, t2, magnitude1)"
        str2 = self.all_transforms[operation2]+"(img, img3, t, t3, magnitude2)"
        img, t = eval(str1)
        img, t = eval(str2)
        return img, t, magnitude_idx1, magnitude_idx2


def CutoutAbs(img, v):  # [0, 20] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img
