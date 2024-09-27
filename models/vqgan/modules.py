import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------- CNN ops -----------
class ConvModule(nn.Module):
    def __init__(self,
                 # ------ Basic conv parameters ------
                 in_dim      :int,
                 out_dim     :int,
                 kernel_size :int = 1,
                 padding     :int = 0,
                 stride      :int = 1,
                ):
        super(ConvModule, self).__init__()
        # ----------- Model parameters -----------
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, stride=stride)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)

        return x

class DeConvModule(nn.Module):
    def __init__(self,
                 # ------ Basic conv parameters ------
                 in_dim      :int,
                 out_dim     :int,
                 kernel_size :int = 1,
                 padding     :int = 0,
                 stride      :int = 1,
                ):
        super(DeConvModule, self).__init__()
        # ----------- Model parameters -----------
        self.conv = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, stride=stride)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)

        return x


# ----------- Attention modules -----------
class NonLocalBlock(nn.Module):
    def __init__(self, in_dim : int):
        super(NonLocalBlock, self).__init__()
        self.in_dim = in_dim

        self.qkv_proj = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.proj_out = nn.Linear(in_dim, in_dim, bias=False)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        shortcut = x
        bs, c, h, w = x.shape

        # [bs, c, h, w] -> [bs, c, hw] -> [bs, hw, c]
        x = x.view(bs, c, -1).permute(0, 2, 1).contiguous()
        x = self.norm(x)

        # QKV projection
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)

        scores = q @ k.transpose(1, 2).contiguous()  # [bs, hw, hw]
        scores = scores * (c**(-0.5))
        scores = F.softmax(scores, dim=-1)

        out = scores @ v  # [bs, hw, c]
        out = self.proj_out(out)

        # [bs, hw, c] -> [bs, c, hw] -> [bs, c, h, w]
        out = out.permute(0, 2, 1).contiguous().view(bs, c, h, w)
        out = out + shortcut

        return out
    
    
# ----------- ResNet modules -----------
class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, shortcut=True):
        super(ResBlock, self).__init__()
        self.shortcut = shortcut and (in_dim == out_dim)
        inter_dim = out_dim // 2
        # ----------------- Network setting -----------------
        self.res_layer = nn.Sequential(
            ConvModule(in_dim,    inter_dim, kernel_size=1, padding=0, stride=1),
            ConvModule(inter_dim, inter_dim, kernel_size=3, padding=1, stride=1),
            ConvModule(inter_dim, out_dim,   kernel_size=1, padding=0, stride=1),
        )

    def forward(self, x):
        h = self.res_layer(x)
        return x + h if self.shortcut else h

class ResStage(nn.Module):
    def __init__(self, in_dim, out_dim, num_blocks=1) -> None:
        super(ResStage, self).__init__()
        # ----------------- Network setting -----------------
        res_blocks = []
        for i in range(num_blocks):
            if i == 0:
                shortcut = (in_dim == out_dim)
                res_blocks.append(ResBlock(in_dim, out_dim, shortcut))
            else:
                res_blocks.append(ResBlock(out_dim, out_dim, True))
        self.res_blocks = nn.Sequential(*res_blocks)

    def forward(self, x):
        return self.res_blocks(x)
