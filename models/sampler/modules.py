import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Type, Tuple


# ----------- Transformer modules -----------
def scaled_dot_product_attention(query, key, value, attn_mask=None):
    """
    :param query: Query 向量 (batch_size, n_heads, seq_len, d_k)
    :param key: Key 向量 (batch_size, n_heads, seq_len, d_k)
    :param value: Value 向量 (batch_size, n_heads, seq_len, d_v)
    :param attn_mask: 注意力掩码 (batch_size, n_heads, seq_len, seq_len)
    :return: 输出向量 (batch_size, n_heads, seq_len, d_v)
    """
    scores = torch.matmul(query, key.transpose(-2, -1))  # (batch_size, n_heads, seq_len, seq_len)
    
    dk = torch.tensor(key.size(-1), dtype=torch.float32)  # d_k
    scores = scores / torch.sqrt(dk)  # 缩放点积
    
    if attn_mask is not None:
        attn_mask_ = attn_mask[:, :, :scores.shape[-2], :scores.shape[-1]]
        scores = scores.masked_fill(attn_mask_ == 0, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)  # (batch_size, n_heads, seq_len, seq_len)
    
    output = torch.matmul(attn_weights, value)  # (batch_size, n_heads, seq_len, d_v)
    
    return output

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # y = x / sqrt(E[x^2] + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
class Attention(nn.Module):
    def __init__(self,
                 dim       :int,
                 num_heads :int   = 8,
                 dropout   :float = 0.,
                 max_seq_len :int = 1024,
                 ):
        super().__init__()
        # --------------- Basic parameters ---------------
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_seq_len = max_seq_len

        # --------------- Network parameters ---------------
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj  = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        # Causal mask: [bs, n_head, seq_len, seq_len]
        self.register_buffer("attn_mask", torch.ones([1, 1, max_seq_len, max_seq_len],
                                                     dtype=torch.bool,
                                                     ).tril(diagonal=0))

    def apply_rotary_emb(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        
        # Reshape for broadcast
        ndim = xq_.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (xq_.shape[1], xq_.shape[-1])

        shape = [d if i == 1 or i == ndim - 1 else 1
                for i, d in enumerate(xq_.shape)]
        
        freqs_cis = freqs_cis.view(*shape)

        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

        return xq_out.type_as(xq), xk_out.type_as(xk)

    def forward(self, x, freqs_cis):
        bs, seq_len, _ = x.shape
        # ----------------- Input proj -----------------
        q, k, v = torch.chunk(self.qkv_proj(x), chunks=3, dim=2)
        ## [bs, seq_len, n_head, c] -> [bs, seq_len, n_head, d_head]
        q = q.view(bs, seq_len, self.num_heads, self.head_dim)
        k = k.view(bs, seq_len, self.num_heads, self.head_dim)
        v = v.view(bs, seq_len, self.num_heads, self.head_dim)

        # Add RoPE
        q, k = self.apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        # [bs, seq_len, n_head, d_head] -> [bs, n_head, seq_len, d_head]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ----------------- Multi-head Attn -----------------
        x = scaled_dot_product_attention(q, k, v, attn_mask=self.attn_mask)

        # ----------------- Output -----------------
        # [bs, n_head, seq_len, d_head] -> [bs, seq_len, n_head, d_head] -> [bs, seq_len, c]
        x = x.permute(0, 2, 1, 3).contiguous().view(bs, seq_len, -1)
        x = self.out_proj(x)
        x = self.proj_drop(x)

        return x

class FFN(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 mlp_dim: int,
                 dropout: float = 0.0,
                 ) -> None:
        super().__init__()
        self.fc1  = nn.Linear(embedding_dim, mlp_dim)
        self.fc2  = nn.Linear(mlp_dim, embedding_dim)
        self.drop = nn.Dropout(dropout)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self,
                 dim         :int,
                 num_heads   :int,
                 mlp_ratio   :float = 4.0,
                 dropout     :float = 0.1,
                 max_seq_len :int = 1024,
                 ) -> None:
        super().__init__()
        # -------------- Model parameters --------------
        self.norm1 = RMSNorm(dim)
        self.attn  = Attention(dim, num_heads, max_seq_len, dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn   = FFN(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        shortcut = x
        # Attention (with prenorm)
        x = self.norm1(x)
        x = self.attn(x, freqs_cis)
        x = shortcut + x

        # Feedforward (with prenorm)
        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = shortcut + x

        return x, freqs_cis