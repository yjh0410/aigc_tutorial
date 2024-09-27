import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Tuple


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

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

    return freqs_cis

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
                 max_seq_len :int = 1024,
                 dropout   :float = 0.,
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
    

# ------------------------ MiniGPT ------------------------
class MiniGPT2(nn.Module):
    def __init__(self,
                 num_layers  : int = 24,
                 num_heads   : int = 16,
                 embed_dim   : int = 1024,
                 max_seq_len : int = 2048,
                 vocab_size  : int = 1024,
                 rope_theta  : int = 50000,
                 ) -> None:
        super().__init__()
        # ----------- Basic parameters -----------
        self.embed_dim   = embed_dim
        self.num_layers  = num_layers
        self.max_seq_len = max_seq_len
        self.vocab_size  = vocab_size
        self.rope_theta  = rope_theta

        # ----------- Model parameters -----------
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.blocks  = nn.ModuleList([
            TransformerBlock(dim = embed_dim,
                             num_heads = num_heads,
                             mlp_ratio = 4.0,
                             dropout   = 0.1,
                             max_seq_len = max_seq_len,
                             )
                             for _ in range(num_layers)])

        self.norm = RMSNorm(self.embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        self._init_weights()
        ## RoPE
        self.freqs_cis = precompute_freqs_cis(
            embed_dim // num_heads,
            max_seq_len * 2,
            rope_theta,
        )

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(self, tok_ids, embeddings=None):
        tok_embed = self.tok_emb(tok_ids)

        if embeddings is not None:  # Prepend explicit embeddings
            tok_embed = torch.cat([embeddings, tok_embed], dim=1)

        seq_len = tok_embed.shape[1]
        assert seq_len <= self.max_seq_len, "Cannot forward, model block size is exhausted."
        
        freqs_cis = self.freqs_cis[:seq_len].to(tok_ids.device)
        for block in self.blocks:
            tok_embed, freqs_cis = block(tok_embed, freqs_cis)

        x = self.norm(tok_embed)
        logits = self.head(x)

        return logits, None
    

# ------------------------ GPT-Sampler ------------------------
class GPTSampler(nn.Module):
    def __init__(self, gpt_config : Dict) -> None:
        super().__init__()
        # ----------- Basic parameters -----------
        self.gpt_config = gpt_config
        self.vocab_size = gpt_config['vocab_size']
        self.sos_token  = gpt_config['sos_token_id']

        # ----------- Model parameters -----------
        self.transformer = MiniGPT2(num_layers  = gpt_config['num_layers'],
                                    num_heads   = gpt_config['num_heads'],
                                    embed_dim   = gpt_config['embed_dim'],
                                    max_seq_len = gpt_config['max_seq_len'],
                                    vocab_size  = gpt_config['vocab_size'],
                                    rope_theta  = gpt_config['rope_theta'],
                                    )

    def compute_loss(self, logits, target):
        """
        logits: [bs, seq_len, c]
        target: [bs, seq_len,]
        """
        loss = F.cross_entropy(logits.flatten(0, 1),  # [BN, vocab_size]
                            target.flatten(),      # [BN,]
                            reduction="mean",
                            )
        loss_dict = {
            'loss': loss,
        }
        
        return loss_dict

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")

        return out

    def sample(self, init_tok_ids, condition, num_steps=1, temperature=1.0, top_k=100):
        self.transformer.eval()
        tok_ids = torch.cat([condition, init_tok_ids], dim=1)
        for k in range(num_steps):
            logits, _ = self.transformer(tok_ids)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)

            probs = F.softmax(logits, dim=-1)

            # Sample a new token id according to the probs
            new_tok_id = torch.multinomial(probs, num_samples=1)

            # Add the new token id
            tok_ids = torch.cat([tok_ids, new_tok_id], dim=1)

        tok_ids = tok_ids[:, condition.shape[1]:].contiguous()

        return tok_ids

    def forward(self, tok_ids):
        # Set SOS token
        sos_tokens = torch.ones(tok_ids.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to(tok_ids.device)

        # Append sos token id
        input_tok_ids = torch.cat((sos_tokens, tok_ids), dim=1)
        logits, _ = self.transformer(input_tok_ids[:, :-1])

        output = {
            'logits': logits,
        }
        if self.training:
            loss_dict = self.compute_loss(logits, target=tok_ids)
            output['loss_dict'] = loss_dict

        return output

def build_gpt_sampler(args, num_vq_embeds=1024):
    if args.sampler == 'gpt_large':
        gpt_config = {
            'num_layers': 32,
            'num_heads': 25,
            'embed_dim': 1600,
            'max_seq_len': 512,
            'rope_theta': 50000,
            'sos_token_id': 0,
            }
    elif args.sampler ==  'gpt_medium':
        gpt_config = {
            'num_layers': 24,
            'num_heads': 16,
            'embed_dim': 1024,
            'max_seq_len': 512,
            'rope_theta': 50000,
            'sos_token_id': 0,
            }
    elif args.sampler ==  'gpt_base':
        gpt_config = {
            'num_layers': 12,
            'num_heads': 12,
            'embed_dim': 768,
            'max_seq_len': 512,
            'rope_theta': 50000,
            'sos_token_id': 0,
            }
    elif args.sampler ==  'gpt_small':
        gpt_config = {
            'num_layers': 10,
            'num_heads': 8,
            'embed_dim': 512,
            'max_seq_len': 512,
            'rope_theta': 50000,
            'sos_token_id': 0,
            }
    else:
        raise NotImplementedError(f"Unknown scale for VQGANSampler: {args.sampler}")

    print(f" - GPT sampler version : {args.sampler}")
    gpt_config["vocab_size"] = num_vq_embeds
    
    return GPTSampler(gpt_config)


if __name__ == '__main__':
    import torch

    # Prepare token ids as the input
    bs, seq_len = 5, 278
    vocab_size = 512
    token_ids = torch.randint(low=0, high=vocab_size, size=[bs, seq_len])

    # Build VQ-VAE sampler
    gpt_config = {
        'num_layers': 12,
        'num_heads': 3,
        'embed_dim': 192,
        'max_seq_len': 512,
        'rope_theta': 50000,
        'sos_token_id': 0,
        'vocab_size': vocab_size
    }
    sampler = GPTSampler(gpt_config)
    sampler.train()

    # Inference
    output = sampler(token_ids)
    for k in output:
        if k == "loss_dict":
            for k_loss in output['loss_dict']:
                print(output['loss_dict'][k_loss])
        else:
            print(f"{k}: ", output[k].shape)