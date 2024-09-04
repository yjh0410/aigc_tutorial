import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, shortcut=True) -> None:
        super(ResBlock, self).__init__()
        self.shortcut = shortcut and (in_dim == out_dim)
        # ----------------- Network setting -----------------
        self.res_layer = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        h = self.res_layer(x)
        return x + h if self.shortcut else h
