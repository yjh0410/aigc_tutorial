import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .encoder  import VqVaeEncoder
    from .codebook import CodeBook
    from .decoder  import VqVaeDecoder
except:
    from  encoder  import VqVaeEncoder
    from  codebook import CodeBook
    from  decoder  import VqVaeDecoder


# ------------------ VQ-VAE ------------------
class VqVAE(nn.Module):
    def __init__(self, img_dim=3, hidden_dim=128, latent_dim=64):
        super().__init__()
        self.encoder  = VqVaeEncoder(img_dim, hidden_dim)
        self.decoder  = VqVaeDecoder(img_dim, hidden_dim, latent_dim)
        self.codebook = CodeBook(hidden_dim, latent_dim, num_embeddings=512)
    
    def compute_loss(self, x, x_rec, vq_output):
        # ----------- Reconstruction loss -----------
        rec_loss = F.mse_loss(x, x_rec, reduction='mean')

        # ----------- Latent loss -----------
        emb_loss = vq_output['emb_loss']
        perplexity = vq_output['perplexity']

        # Total loss
        loss = rec_loss + emb_loss
        loss_dict = {
            'rec_loss': rec_loss,
            'emb_loss': emb_loss,
            'loss': loss,
            'perplexity': perplexity
        }

        return loss_dict
    
    def forward_encode(self, x):
        # Encode
        z_e = self.encoder(x)

        # Quantize
        vq_output = self.codebook(z_e)

        return vq_output['rep_z_q']
    
    def forward_decode(self, z_q):
        x_rec = self.decoder(z_q)
        return x_rec
    
    def forward(self, x):
        # Encode
        z_e = self.encoder(x)

        # Quantize
        vq_output = self.codebook(z_e)

        # Decode
        x_rec = self.decoder(vq_output['rep_z_q'])

        output = {
            'x_pred': x_rec,
        }

        # Compute loss
        if self.training:
            loss_dict = self.compute_loss(x, x_rec, vq_output)
            output['loss_dict'] = loss_dict

        return output
    

if __name__ == '__main__':
    import torch
    from thop import profile

    # Prepare an image as the input
    bs, img_dim, img_size = 4, 3, 128
    hidden_dim = 256
    latent_dim = 128
    x = torch.randn(bs, img_dim, img_size, img_size)

    # Build model
    model = VqVAE(img_dim, hidden_dim, latent_dim)
    model.train()

    # Inference
    outputs = model(x)
    loss_dict = outputs.pop("loss_dict")
    for k in loss_dict:
            print(f"{k}: ", loss_dict[k].item())
    for k in outputs:
            print(f"{k}: ", outputs[k].shape)

    # Compute FLOPs & Params
    print('==============================')
    model.eval()
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))

