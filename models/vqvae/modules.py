import torch
import torch.nn as nn
from typing import Type


# ----------- CNN modules -----------
class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, shortcut=True) -> None:
        super(ResBlock, self).__init__()
        self.shortcut = shortcut and (in_dim == out_dim)
        inter_dim = out_dim // 2
        # ----------------- Network setting -----------------
        self.res_layer = nn.Sequential(
            nn.Conv2d(in_dim, inter_dim, kernel_size=1, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_dim, inter_dim, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_dim, out_dim, kernel_size=1, padding=0, stride=1),
            nn.ReLU(inplace=True),
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


# ----------- Transformer modules -----------
class Attention(nn.Module):
    def __init__(self,
                 dim       :int,
                 num_heads :int   = 8,
                 dropout   :float = 0.
                 ):
        super().__init__()
        # --------------- Basic parameters ---------------
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # --------------- Network parameters ---------------
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj  = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, q, k, v):
        assert k.shape == v.shape
        bs, Nq, _ = q.shape
        bs, Nm, _ = k.shape
        # ----------------- Input proj -----------------
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # ----------------- Multi-head Attn -----------------
        ## [B, N, C] -> [B, N, H, C_h] -> [B, H, N, C_h]
        q = q.view(bs, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.view(bs, Nm, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.view(bs, Nm, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        ## [B, H, Nq, C_h] X [B, H, C_h, Nk] = [B, H, Nq, Nk]
        attn = q * self.scale @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v # [B, H, Nq, C_h]

        # ----------------- Output -----------------
        x = x.permute(0, 2, 1, 3).contiguous().view(bs, Nq, -1)
        x = self.out_proj(x)
        x = self.proj_drop(x)

        return x

class FFN(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 mlp_dim: int,
                 act: Type[nn.Module] = nn.GELU,
                 dropout: float = 0.0,
                 ) -> None:
        super().__init__()
        self.fc1   = nn.Linear(embedding_dim, mlp_dim)
        self.drop1 = nn.Dropout(dropout)
        self.fc2   = nn.Linear(mlp_dim, embedding_dim)
        self.drop2 = nn.Dropout(dropout)
        self.act   = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self,
                 dim       :int,
                 num_heads :int,
                 mlp_ratio :float = 4.0,
                 act_layer :Type[nn.Module] = nn.GELU,
                 dropout   :float = 0.
                 ) -> None:
        super().__init__()
        # -------------- Model parameters --------------
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = FFN(dim, int(dim * mlp_ratio), act_layer, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        # Attention (with prenorm)
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x

        # Feedforward (with prenorm)
        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = shortcut + x

        return x
