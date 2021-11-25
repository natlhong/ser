import pytest

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

import torch
from torchvision import transforms as torch_transforms

im = Image.open('data/Image.png').convert('L')
im_gray = np.array(im)
im_test = np.flip(im_gray, axis = 0)
im_test = np.flip(im_test, axis = 1)

def flip():
    """
    Flip a tensor both vertically and horizontally
    """
    return torch_transforms.Compose(
        [
            torch_transforms.RandomHorizontalFlip(p=1.0),
            torch_transforms.RandomVerticalFlip(p=1.0),
        ]
    )

def test_flip():
    ts = flip()
    assert np.array_equal(np.array(ts(im)),im_test)