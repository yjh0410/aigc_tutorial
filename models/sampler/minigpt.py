import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .modules  import TransformerBlock
except:
    from  modules  import TransformerBlock


# ------------------------ Basic Modules ------------------------
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

    return freqs_cis


class MiniGPT2(nn.Module):
    def __init__(self,
                 num_layers  : int = 24,
                 num_heads   : int = 16,
                 embed_dim   : int = 1024,
                 max_seq_len : int = 2048,
                 vocab_size  : int = 1024,
                 rope_theta  : int = 50000,
                 dropout     : float = 0.0,
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
        self.transformers = nn.ModuleList([
            TransformerBlock(dim = embed_dim,
                             num_heads = num_heads,
                             mlp_ratio = 4.0,
                             act_layer = nn.GELU,
                             dropout = dropout,
                             max_seq_len = max_seq_len,
                             )
                             for _ in range(num_layers)])

        self.norm_layer = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        ## RoPE
        self.freqs_cis = precompute_freqs_cis(
            embed_dim // num_heads,
            max_seq_len * 2,
            rope_theta,
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(self, token_ids, embeddings=None):
        tok_embed = self.tok_emb(token_ids)

        if embeddings is not None:  # Prepend explicit embeddings
            tok_embed = torch.cat([embeddings, tok_embed], dim=1)

        seq_len = tok_embed.shape[1]
        assert seq_len <= self.max_seq_len, "Cannot forward, model block size is exhausted."
        
        freqs_cis = self.freqs_cis[:seq_len].to(token_ids.device)
        for block in self.transformers:
            tok_embed, freqs_cis = block(tok_embed, freqs_cis)

        x = self.norm_layer(tok_embed)
        logits = self.head(x)

        return logits, None
    

if __name__ == "__main__":
    # Prepare token ids as the input
    bs, seq_len = 5, 278
    vocab_size = 512
    token_ids = torch.randint(low=0, high=512, size=[bs, seq_len])

    # Build model
    sampler = MiniGPT2(num_layers  = 3,
                       num_heads   = 2,
                       dropout     = 0.1,
                       embed_dim   = 128,
                       vocab_size  = vocab_size,
                       max_seq_len = 1280,
                       )
    sampler.train()

    # Inference
    logits, _ = sampler(token_ids)
    print(logits.shape)
    