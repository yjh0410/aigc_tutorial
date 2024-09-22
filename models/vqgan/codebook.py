import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------ VQ-GAN Codebooks ------------------
class CodeBook(nn.Module):
    def __init__(self, num_embeddings=512, hidden_dim=256, latent_dim=128):
        super(CodeBook, self).__init__()
        # ---------- Basic parameters ----------
        self.ema_decay = 0.99
        self.latent_dim = latent_dim      # D defined in paper
        self.num_embeddings = num_embeddings  # K defined in paper

        # ---------- Model parameters ----------
        self.input_proj = nn.Conv2d(hidden_dim, latent_dim, kernel_size=1)
        self.embedding  = nn.Embedding(num_embeddings, latent_dim)  # [K, D]
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

        print(" Use EMA trick for VQ-VAE ...")
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embedding_ema", self.embedding.weight.clone())

    def forward(self, z):
        # Input projection
        z = self.input_proj(z)
        z = z.permute(0, 2, 3, 1).contiguous()    # [B, C, H, W] -> [B, H, W, C]
        z_flattened = z.view(-1, self.latent_dim) # [B, H, W, C] -> [BHW, C]

        # Distance between image feature and all embeddings
        dist = torch.cdist(z_flattened, self.embedding.weight, p=2) ** 2  # [BHW, K]
        
        # Find closest encodings
        min_indices = torch.argmin(dist, dim=1)  # [BHW,]
        min_indices_ot = F.one_hot(min_indices, num_classes=self.num_embeddings)  # [BHW, K]
        
        # Index latent vectors
        z_q = self.embedding(min_indices)  # [BHW, C]
        z_q = z_q.view(z.shape)            # [BHW, C] -> [B, H, W, C]

        # Preserve gradients
        rep_z_q = z + (z_q - z).detach()

        # [B, H, W, C] -> [B, C, H, W]
        rep_z_q = rep_z_q.permute(0, 3, 1, 2).contiguous()

        vq_output = {
            'rep_z_q': rep_z_q,
            'min_encodings': min_indices,
        }

        # --------------- Loss of Vector-quantizer ---------------
        if self.training:
            # MSE loss between Z_q and Z_E
            emb_loss = F.mse_loss(z_q.detach(), z, reduction='mean')

            # EMA update cluster size
            cur_cluster_size = torch.sum(min_indices_ot, dim=0)  # [BHW, K] -> [K,], cluster size for each embed
            self.cluster_size.data.mul_(self.ema_decay).add_(
                cur_cluster_size, alpha=1 - self.ema_decay
            )

            # EMA update embeds
            embed_sum = min_indices_ot.transpose(0, 1).float() @  z_flattened  #[K, BHW] x [BHW, C] = [K, C]
            self.embedding_ema.data.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)

            # Normalized embeddings
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
            embed_normalized = self.embedding_ema / cluster_size.unsqueeze(1)

            # Updata codebook with EMA result
            self.embedding.weight.data.copy_(embed_normalized)

            # Perplexity
            e_mean = torch.mean(min_indices_ot.float(), dim=0)
            perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

            vq_output['emb_loss'] = emb_loss
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
    print("rep_z_q: ",    vq_output['rep_z_q'].shape)
    print("emb_loss: ",   vq_output['emb_loss'])

    # Compute FLOPs & Params
    print('==============================')
    model.eval()
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))

