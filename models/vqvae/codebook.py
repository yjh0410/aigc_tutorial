import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------ VQ-VAE Modules ------------------
class CodeBook(nn.Module):
    def __init__(self, hidden_dim=256, latent_dim=128, num_embeddings=512):
        super(CodeBook, self).__init__()
        # ---------- Basic parameters ----------
        self.latent_dim     = latent_dim      # D defined in paper
        self.num_embeddings = num_embeddings  # K defined in paper

        # ---------- Model parameters ----------
        self.embedding  = nn.Embedding(num_embeddings, latent_dim)  # [K, D]
        self.input_proj = nn.Conv2d(hidden_dim, latent_dim, kernel_size=1)

        # Initialize all layers
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z):
        # Input projection
        z = self.input_proj(z)

        # [B, C, H, W] -> [B, H, W, C] -> [BHW, C]
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        # Distance between image feature and all embeddings
        dist = torch.cdist(z_flattened, self.embedding.weight, p=2) ** 2  # [BHW, K]

        # Find closest encodings
        min_encoding_indices = torch.argmin(dist, dim=1).unsqueeze(1)

        # One-hot format including the index of the closest encodings
        min_encodings = torch.zeros(min_encoding_indices.shape[0],
                                    self.num_embeddings,
                                    ).to(z.device) # [BHW, K]
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # Quantized latent vectors: [BHW, K] x [K, C] = [BHW, C]
        z_q = torch.matmul(min_encodings, self.embedding.weight)
        z_q = z_q.view(z.shape) # [BHW, C] -> [B, H, W, C]

        # Preserve gradients
        rep_z_q = z + (z_q - z).detach()

        # [B, H, W, C] -> [B, C, H, W]
        rep_z_q = rep_z_q.permute(0, 3, 1, 2).contiguous()

        vq_output = {
            'rep_z_q': rep_z_q,
        }

        # --------------- Loss of Vector-quantizer ---------------
        if self.training:
            # Embedding loss
            beta = 0.25
            embedding_loss = F.mse_loss(z_q.detach(), z, reduction='mean') + \
                             F.mse_loss(z.detach(), z_q, reduction='mean') * beta
            # Perplexity
            e_mean = torch.mean(min_encodings, dim=0)
            perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

            vq_output['emb_loss'] = embedding_loss
            vq_output['perplexity'] = perplexity

        return vq_output
    

if __name__ == '__main__':
    import torch
    from thop import profile

    # Prepare an image as the input
    bs, latent_dim, img_size = 2, 256, 32
    x = torch.randn(bs, latent_dim, img_size, img_size)

    # Build model
    model = CodeBook(num_embeddings=512, latent_dim=256)
    model.train()

    # Inference
    vq_output = model(x)
    for k in vq_output:
        if k is not None:
            print(f"{k}: ", vq_output[k].shape)

    # Compute FLOPs & Params
    print('==============================')
    model.eval()
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))

