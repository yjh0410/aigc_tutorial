import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------- CNN ops -----------
class ConvModule(nn.Module):
    def __init__(self,
                 in_dim      :int,
                 out_dim     :int,
                 kernel_size :int = 1,
                 padding     :int = 0,
                 stride      :int = 1,
                ):
        super(ConvModule, self).__init__()
        # ----------- Model parameters -----------
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.norm = nn.GroupNorm(num_groups=32, num_channels=out_dim)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        # PreNorm and PreAct
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x

class DeConvModule(nn.Module):
    def __init__(self,
                 in_dim      :int,
                 out_dim     :int,
                 kernel_size :int = 1,
                 padding     :int = 0,
                 stride      :int = 1,
                ):
        super(DeConvModule, self).__init__()
        # ----------- Model parameters -----------
        self.conv = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.norm = nn.GroupNorm(num_groups=32, num_channels=out_dim)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        # PreNorm and PreAct
        x = self.conv(x)
        x = self.norm(x)
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
    def __init__(self, in_dim, out_dim):
        super(ResBlock, self).__init__()
        # ----------------- Network setting -----------------
        self.res_layer = nn.Sequential(
            ConvModule(in_dim,  out_dim, kernel_size=3, padding=1, stride=1),
            ConvModule(out_dim, out_dim, kernel_size=3, padding=1, stride=1),
        )
        if in_dim != out_dim:
            self.identity = nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, stride=1)
        else:
            self.identity = nn.Identity()

    def forward(self, x):
        return self.res_layer(x) + self.identity(x)

class ResStage(nn.Module):
    def __init__(self, in_dim, out_dim, num_blocks=1, use_attn=False):
        super(ResStage, self).__init__()
        # ----------------- Network setting -----------------
        res_blocks = []
        for i in range(num_blocks):
            if i == 0:
                res_blocks.append(ResBlock(in_dim, out_dim))
            else:
                res_blocks.append(ResBlock(out_dim, out_dim))
            if use_attn:
                res_blocks.append(NonLocalBlock(out_dim))
        self.res_blocks = nn.Sequential(*res_blocks)

    def forward(self, x):
        return self.res_blocks(x)
