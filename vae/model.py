import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------- CNN Modules --------------
class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, shortcut=True) -> None:
        super(ResBlock, self).__init__()
        self.shortcut = shortcut and (in_dim == out_dim)
        # ----------------- Network setting -----------------
        self.res_layer = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        h = self.res_layer(x)
        return x + h if self.shortcut else h


# -------------- VAE Encoder --------------
class VaeEncoder(nn.Module):
    def __init__(self, img_dim=3, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
        # ---------- Model parameters ----------
        self.layer_1 = nn.Sequential(
            nn.Conv2d(img_dim, 64, kernel_size=2, padding=0, stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, padding=0, stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer_3 = ResBlock(128, 128, shortcut=True)
        self.layer_4 = ResBlock(128, 128, shortcut=True)
        self.layer_5 = nn.Conv2d(128, 2 * latent_dim, kernel_size=1, padding=0, stride=1)
        
        # Initialize all layers
        self.init_weights()

    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.reset_parameters()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        
        mu, log_var = torch.chunk(x, 2, dim=1)

        return mu, log_var


# -------------- VAE Decoder --------------
class VaeDecoder(nn.Module):
    def __init__(self, img_dim=3, latent_dim=8):
        super().__init__()
        # ---------- Basic parameters ----------
        self.latent_dim = latent_dim
        # ---------- Model parameters ----------
        self.layer_1 = nn.Sequential(
            nn.Conv2d(latent_dim, 128, kernel_size=1, padding=0, stride=1),
            nn.ReLU(inplace=True),
        )
        self.layer_2 = ResBlock(128, 128, shortcut=True)
        self.layer_3 = ResBlock(128, 128, shortcut=True)
        self.layer_4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, padding=0, stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer_5 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, padding=0, stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer_6 = nn.Conv2d(64, img_dim, kernel_size=1, padding=0, stride=1)

        # Initialize all layers
        self.init_weights()

    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.reset_parameters()

    def forward(self, x):
        # From latent space into image space: [B, Z] -> [B, N] -> [B, C, H, W]
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)

        return x
    

# -------------- Variational AutoEncoder --------------
class VAE(nn.Module):
    def __init__(self, img_dim=3, latent_dim=8):
        super().__init__()
        self.num_samples = 1
        self.latent_dim  = latent_dim
        self.encoder = VaeEncoder(img_dim, latent_dim)
        self.decoder = VaeDecoder(img_dim, latent_dim)

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
    
    def forward_encode(self, x):
        """
            Input:
                x: (torch.Tensor) -> [B, 3, H, W], the input image tensor.
        """
        mu, log_var = self.encoder(x)

        return mu, log_var
    
    def forward_decode(self, rep_z):
        """
            Input:
                rep_z: (torch.Tensor) -> [B, C, H, W], the latent tensor.
        """
        x_rec = self.decoder(rep_z)

        return x_rec
    
    def forward(self, x):
        # --------- Encode ---------
        mu, log_var = self.forward_encode(x)

        # --------- Decode ---------
        num_samples = self.num_samples if self.training else 1
        x_rec = []
        for i in range(num_samples):
            rep_z = self.reparameterization(mu, log_var)
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

    print(' \n=========== VAE Encoder =========== ')
    # Prepare an image as the input
    bs, img_dim, img_size = 2, 3, 128
    latent_dim = 8
    x = torch.randn(bs, img_dim, img_size, img_size)

    # Build model
    model = VaeEncoder(latent_dim=latent_dim)

    # Inference
    z = model(x)

    # Compute FLOPs & Params
    model.eval()
    x = torch.randn(1, img_dim, img_size, img_size)
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('Encoder FLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Encoder Params : {:.2f} M'.format(params / 1e6))


    print(' \n=========== VAE Decoder =========== ')
    # Prepare an image as the input
    x = torch.randn(bs, latent_dim, img_size // 4, img_size // 4)

    # Build model
    model = VaeDecoder(latent_dim=8)

    # Inference
    outputs = model(x)
    print(outputs.shape)

    # Compute FLOPs & Params
    model.eval()
    x = torch.randn(1, latent_dim, img_size // 4, img_size // 4)
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('Decoder FLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Decoder Params : {:.2f} M'.format(params / 1e6))


    print(' \n=========== VAE =========== ')
    # Prepare an image as the input
    bs, img_dim, img_size = 2, 3, 128
    x = torch.randn(bs, img_dim, img_size, img_size)

    # Build model
    model = VAE()

    # Inference
    outputs = model(x)
    if "loss_dict" in outputs:
        loss_dict = outputs['loss_dict']
        for k in loss_dict:
            print(k, loss_dict[k].item())

    # Compute FLOPs & Params
    model.eval()
    x = torch.randn(1, img_dim, img_size, img_size)
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('VAE GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('VAE Params : {:.2f} M'.format(params / 1e6))
