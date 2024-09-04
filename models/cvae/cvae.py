import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .encoder import CVaeEncoder
    from .decoder import CVaeDecoder
except:
    from  encoder import CVaeEncoder
    from  decoder import CVaeDecoder


# ------------------ Variational AutoEncoder ------------------
class CondiontalVAE(nn.Module):
    def __init__(self, img_dim=3, latent_dim=8):
        super().__init__()
        self.num_samples = 1
        self.latent_dim  = latent_dim
        self.encoder = CVaeEncoder(img_dim, latent_dim)
        self.decoder = CVaeDecoder(img_dim, latent_dim)

    def reparameterization(self, mu, log_var):
        # sample from N(0, 1) distribution
        z = torch.randn_like(log_var)

        # Reparam: rep_z = \mu + \sigma * z
        rep_z = mu + z * torch.sqrt(log_var.exp())

        return rep_z
    
    def compute_loss(self, x, output):
        bs = x.shape[0]
        # ----------- Reconstruction loss -----------
        x_rec_list = output['x_pred']
        num_samples = len(x_rec_list)
        rec_loss = []
        for i in range(num_samples):            
            rec_loss_i = F.mse_loss(x, x_rec_list[i], reduction='none')
            rec_loss.append(rec_loss_i.sum() / bs)
        rec_loss = sum(rec_loss) / num_samples

        # ----------- Latent loss -----------
        mu_pred = output['mu_pred']
        log_var_pred = output['log_var_pred']
        latent_loss = 0.5 * (-1.0 + log_var_pred.exp() + torch.square(mu_pred) - log_var_pred)
        latent_loss = latent_loss.sum() / bs

        # Total loss
        loss = rec_loss + latent_loss
        loss_dict = {
            'rec_loss': rec_loss,
            'kl_loss': latent_loss,
            'loss': loss
        }

        return loss_dict
    
    def forward_encode(self, x, label):
        """
            Input:
                x: (torch.Tensor) -> [B, 3, H, W], the input image tensor.
        """
        mu, log_var = self.encoder(x, label)

        return mu, log_var
    
    def forward_decode(self, rep_z):
        """
            Input:
                rep_z: (torch.Tensor) -> [B, C, H, W], the latent tensor.
        """
        x_rec = self.decoder(rep_z)

        return x_rec
    
    def forward(self, x, label):
        # --------- Encode ---------
        mu, log_var = self.forward_encode(x, label)

        # --------- Decode ---------
        num_samples = self.num_samples if self.training else 1
        x_rec = []
        for i in range(num_samples):
            rep_z = self.reparameterization(mu, log_var)
            # Add condition into z
            c = torch.ones([rep_z.shape[0], 1, rep_z.shape[2], rep_z.shape[3]], device=rep_z.device) \
                * label[:, None, None, None]
            rep_z = torch.cat([rep_z, c], dim=1)

            x_rec.append(self.forward_decode(rep_z))
        output = {
            'x_pred':       x_rec,
            'mu_pred':      mu,
            'log_var_pred': log_var,
        }

        if self.training:
            loss_dict = self.compute_loss(x, output)
            output['loss_dict'] = loss_dict
        else:
            output = {
                'x_pred':       x_rec[0],  # only return one sample
                'mu_pred':      mu,
                'log_var_pred': log_var,
            }

        return output
    

if __name__ == '__main__':
    import torch
    from thop import profile

    # Prepare an image as the input
    bs, img_dim, img_size = 2, 3, 128
    x = torch.randn(bs, img_dim, img_size, img_size)
    label = torch.randint(0, 10, [bs])

    # Build model
    model = CondiontalVAE()

    # Inference
    outputs = model(x, label)
    if "loss_dict" in outputs:
        loss_dict = outputs['loss_dict']
        for k in loss_dict:
            print(k, loss_dict[k].item())

    # Compute FLOPs & Params
    print('==============================')
    model.eval()
    flops, params = profile(model, inputs=(x, label, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))

