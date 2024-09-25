# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""

import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter


# IN1K-Cls pretrained weights
model_urls = {
    'resnet18':  torchvision.models.resnet.ResNet18_Weights,
    'resnet34':  torchvision.models.resnet.ResNet34_Weights,
    'resnet50':  torchvision.models.resnet.ResNet50_Weights,
    'resnet101': torchvision.models.resnet.ResNet101_Weights,
}


# -------------------- ResNet series --------------------
class ResNet(nn.Module):
    """Standard ResNet backbone."""
    def __init__(self, name :str = "resnet50", use_pretrained :bool = False):
        super().__init__()
        # Pretrained
        if use_pretrained:
            pretrained_weights = model_urls[name].IMAGENET1K_V1
        else:
            pretrained_weights = None

        # Backbone
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, False],
            norm_layer=nn.BatchNorm2d, weights=pretrained_weights)
        return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.feat_dims = [64, 128, 256, 512] if name in ('resnet18', 'resnet34') else [256, 512, 1024, 2048]

    def forward(self, x):
        xs = self.body(x)
        fmp_list = []
        for name, fmp in xs.items():
            fmp_list.append(fmp)

        return fmp_list


if __name__ == '__main__':
    model = ResNet("resnet50", True)

    x = torch.randn(2, 3, 320, 320)
    output = model(x)
    for y in output:
        print(y.size())