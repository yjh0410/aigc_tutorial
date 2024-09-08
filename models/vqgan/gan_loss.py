import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------- LIPIPS loss --------------
## TODO:


# -------------- DIscrimination loss --------------
## TODO:


class GANLoss(object):
    def __init__(self) -> None:
        pass
        self.pixel_mean = []
        self.pixel_std  = []

    def preprocess_images(self, images):
        images_norm = (images - self.pixel_mean) / self.pixel_std
        return images_norm
    
    