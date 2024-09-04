import math
import random
import torch
import torch.nn as nn
from typing import Type

try:
    from .modules  import TransformerBlock
except:
    from  modules import TransformerBlock


# ------------------------ Basic Modules ------------------------
class VqVAESampler(nn.Module):
    def __init__(self,
                 num_blocks: int = 6,
                 num_heads:  int = 2,
                 dropout:    float = 0.0,
                 latent_dim: int = 128,
                 num_embeddings: int = 512,
                 ) -> None:
        super().__init__()
        # ----------- Basic parameters -----------
        self.latent_dim     = latent_dim
        self.num_embeddings = num_embeddings
        # ----------- Model parameters -----------
        self.mask_tokens  = nn.Parameter(torch.zeros(1, 1, self.latent_dim))
        self.norm_layer   = nn.LayerNorm(self.latent_dim)
        self.transformers = nn.ModuleList([
            TransformerBlock(self.latent_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_blocks)])
        self.out_proj = nn.Linear(latent_dim, num_embeddings)
        
        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # we use xavier_uniform following official JAX ViT:
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def get_posembed(self, embed_dim, num_seq, temperature=10000):
        scale = 2 * math.pi
        num_pos_feats = embed_dim // 2

        # sequence indexes
        indexs =  torch.arange(num_seq, dtype=torch.float32)
        indexs = indexs / (num_seq + 1e-6) * scale
    
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        dim_t_ = torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats
        dim_t = temperature ** (2 * dim_t_)

        pos = torch.div(indexs[..., None], dim_t)
        pos_embed = torch.stack((pos[..., 0::2].sin(),
                                 pos[..., 1::2].cos()), dim=-1).flatten(-2)

        return pos_embed

    def random_masking(self, x):
        B, N, C = x.shape
        len_keep = int(N * (1 - self.mask_ratio))

        noise = torch.rand(B, N, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)        # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # restore the original position of each patch

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get th binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def compute_loss(self, pred_scores, target_ids):
        """
        imgs: [B, 3, H, W]
        pred: [B, N, C], C = p*p*3
        mask: [B, N], 0 is keep, 1 is remove, 
        """
        # TODO: compute ce loss
        loss = None
        
        return loss

    def forward(self, tokens, token_ids):
        bs, seq_len, c = tokens.shape
        # masking: length -> length * mask_ratio
        x_keep, mask, ids_restore = self.random_masking(tokens)

        # Append mask tokens
        num_mask = seq_len - x_keep.shape[1]
        mask_tokens = self.mask_tokens.repeat(bs, num_mask, 1)
        x_all = torch.cat([x_keep, mask_tokens], dim=1)
        x_all = torch.gather(x_all, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, c))  # unshuffle

        # add pos embed
        pos_embed = self.get_posembed(c, seq_len)
        x_all = x_all + pos_embed.to(x.device)

        # Apply Transformer blocks
        for block in self.transformers:
            x = block(x)
        x = self.norm_layer(x)
        pred_scores = None

        if self.training:
            # TODO: training loss
            target_ids = token_ids
            loss = self.compute_loss(pred_scores, target_ids)
        
        return x_all


if __name__ == '__main__':
    import torch

    # Prepare an image as the input
    bs, seq_len, c  = 4, 1024, 128
    num_embeddings = 512
    token = torch.randn(bs, seq_len, c)
    token_ids = torch.randint(low=0, high=512, size=[bs, seq_len])

    # Build model
    sampler = VqVAESampler(num_blocks=3,
                           num_heads=2,
                           dropout=0.1,
                           latent_dim=c,
                           num_embeddings=num_embeddings,
                           )
    sampler.train()

    # Inference
    output = sampler(token, token_ids)
