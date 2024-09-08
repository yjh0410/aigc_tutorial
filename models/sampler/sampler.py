import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict

try:
    from .minigpt import MiniGPT2
except:
    from  minigpt import MiniGPT2


# ------------------------ Basic Modules ------------------------
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
        # Append the sos token id
        sos_tokens = torch.ones(tok_ids.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to(tok_ids.device)

        # Append sos token id
        input_tok_ids = torch.cat((sos_tokens, tok_ids), dim=1)
        logits, _ = self.transformer(input_tok_ids[:, :-1])

        # # Randomly mask token ids
        # mask = torch.bernoulli(0.5 * torch.ones(tok_ids.shape, device=tok_ids.device))
        # mask = mask.round().to(dtype=torch.int64)
        # random_tok_ids = torch.randint_like(tok_ids, self.vocab_size)
        # new_tok_ids = mask * tok_ids + (1 - mask) * random_tok_ids

        # # Append sos token id
        # new_tok_ids = torch.cat((sos_tokens, new_tok_ids), dim=1)
        # logits, _ = self.transformer(new_tok_ids[:, :-1])

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