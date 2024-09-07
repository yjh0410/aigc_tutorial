import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict

try:
    from .minigpt import MiniGPT2
except:
    from  minigpt import MiniGPT2


# ------------------------ Basic Modules ------------------------
class VqVaeSampler(nn.Module):
    def __init__(self,
                 gpt_config    : Dict,
                 num_vq_embeds : int,
                 ) -> None:
        super().__init__()
        gpt_config['vocab_size'] = num_vq_embeds
        # ----------- Basic parameters -----------
        self.gpt_config = gpt_config
        self.num_vq_embeds = num_vq_embeds
        self.vocab_size = gpt_config['vocab_size']
        self.sos_token  = gpt_config['sos_token_id']

        # ----------- Model parameters -----------
        self.transformers = MiniGPT2(num_layers  = gpt_config['num_layers'],
                                     num_heads   = gpt_config['num_heads'],
                                     embed_dim   = gpt_config['embed_dim'],
                                     max_seq_len = gpt_config['max_seq_len'],
                                     vocab_size  = gpt_config['vocab_size'],
                                     rope_theta  = gpt_config['rope_theta'],
                                     dropout     = 0.1,
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

    def load_vqvae_model(self, vqvae_model):
        self.vqvae_model = vqvae_model.eval()

    def encode_to_z(self, x):
        z_q, tok_ids = self.vqvae_model.forward_encode(x)
        return z_q, tok_ids

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")

        return out

    def sample(self, init_tok_ids, condition, num_steps=1, temperature=1.0, top_k=100):
        tok_ids = torch.cat([init_tok_ids, condition], dim=1)
        for k in range(num_steps):
            logits, _ = self.transformers(tok_ids)
            nt_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                nt_logits = self.top_k_logits(nt_logits, top_k)

            probs = F.softmax(nt_logits, dim=-1)

            # Sample a new token id according to the probs
            new_tok_id = torch.multinomial(probs, num_samples=1)

            # Add the new token id
            tok_ids = torch.cat([tok_ids, new_tok_id], dim=1)

        tok_ids = tok_ids[:, condition.shape[1]:].contiguous()

        return tok_ids

    def forward(self, x):

        # Encode to latent space
        with torch.no_grad():
            self.vqvae_model.eval()
            _, tok_ids = self.encode_to_z(x)

        # Prepare the sos token id
        sos_tokens = torch.ones(tok_ids.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to(tok_ids.device)

        # Randomly mask token ids
        mask = torch.bernoulli(0.5 * torch.ones(tok_ids.shape, device=tok_ids.device))
        mask = mask.round().to(dtype=torch.int64)
        random_tok_ids = torch.randint_like(tok_ids, self.vocab_size)
        new_tok_ids = mask * tok_ids + (1 - mask) * random_tok_ids

        # Append sos token id
        new_tok_ids = torch.cat((sos_tokens, new_tok_ids), dim=1)
        logits, _ = self.transformers(new_tok_ids[:, :-1])

        output = {
            'logits': logits,
        }
        if self.training:
            loss_dict = compute_loss(logits, target=tok_ids)
            output['loss_dict'] = loss_dict

        return output


# --------------------- Loss functions ---------------------
def compute_loss(logits, target):
    """
    pred_scores: [bs, seq_len, c]
    target_ids:  [bs, seq_len,]
    """
    loss = F.cross_entropy(logits.flatten(0, 1),  # [BN, vocab_size]
                           target.flatten(),      # [BN,]
                           reduction="mean",
                           )
    loss_dict = {
        'loss': loss,
    }
    
    return loss_dict


if __name__ == '__main__':
    import torch
    from vqvae import VqVAE

    # Prepare an image as the input
    bs, c, h, w  = 4, 3, 64, 64
    num_vq_embeds = 768
    x = torch.randn(bs, c, h, w)

    # Build VQ-VAE model
    vqvae_model = VqVAE(
        img_dim=c,
        hidden_dim=128,
        num_embeddings=num_vq_embeds,
        latent_dim=64)

    # Build VQ-VAE sampler
    gpt_config = {
        'num_layers': 12,
        'num_heads': 3,
        'embed_dim': 192,
        'max_seq_len': 512,
        'rope_theta': 50000,
        'sos_token_id': 0,
    }
    sampler = VqVaeSampler(gpt_config, num_vq_embeds)
    sampler.train()
    sampler.load_vqvae_model(vqvae_model)

    # Inference
    output = sampler(x)
    for k in output:
        if k == "loss_dict":
            for k_loss in output['loss_dict']:
                print(output['loss_dict'][k_loss])
        else:
            print(f"{k}: ", output[k].shape)