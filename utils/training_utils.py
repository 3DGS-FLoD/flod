import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from utils.loss_utils import l1_loss, ssim, ssim_error_map
from utils.image_utils import psnr

import torch
from gaussian_renderer import render
from matplotlib import cm


def expand_list_to_match_lods(lst, lods):
    length = len(lst)
    repeats = lods - length
    return lst + [lst[-1]] * repeats
