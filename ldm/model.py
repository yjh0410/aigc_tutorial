import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Type, Tuple


# ----------- CNN ops -----------
class ConvModule(nn.Module):
    def __init__(self,
                 # ------ Basic conv parameters ------
                 in_dim      :int,
                 out_dim     :int,
                 kernel_size :int = 1,
                 padding     :int = 0,
                 stride      :int = 1,
                ):
        super(ConvModule, self).__init__()
        # ----------- Model parameters -----------
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, stride=stride)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)

        return x

class DeConvModule(nn.Module):
    def __init__(self,
                 # ------ Basic conv parameters ------
                 in_dim      :int,
                 out_dim     :int,
                 kernel_size :int = 1,
                 padding     :int = 0,
                 stride      :int = 1,
                ):
        super(DeConvModule, self).__init__()
        # ----------- Model parameters -----------
        self.conv = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, stride=stride)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)

        return x


# ----------- Attention modules -----------
class NonLocalBlock(nn.Module):
    def __init__(self, in_dim : int):
        super(NonLocalBlock, self).__init__()
        self.in_dim = in_dim

        self.qkv_proj = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.proj_out = nn.Linear(in_dim, in_dim, bias=False)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        shortcut = x
        bs, c, h, w = x.shape

        # [bs, c, h, w] -> [bs, c, hw] -> [bs, hw, c]
        x = x.view(bs, c, -1).permute(0, 2, 1).contiguous()
        x = self.norm(x)

        # QKV projection
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)

        scores = q @ k.transpose(1, 2).contiguous()  # [bs, hw, hw]
        scores = scores * (c**(-0.5))
        scores = F.softmax(scores, dim=-1)

        out = scores @ v  # [bs, hw, c]
        out = self.proj_out(out)

        # [bs, hw, c] -> [bs, c, hw] -> [bs, c, h, w]
        out = out.permute(0, 2, 1).contiguous().view(bs, c, h, w)
        out = out + shortcut

        return out
    
    
# ----------- ResNet modules -----------
class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, shortcut=True):
        super(ResBlock, self).__init__()
        self.shortcut = shortcut and (in_dim == out_dim)
        inter_dim = out_dim // 2
        # ----------------- Network setting -----------------
        self.res_layer = nn.Sequential(
            ConvModule(in_dim,    inter_dim, kernel_size=1, padding=0, stride=1),
            ConvModule(inter_dim, inter_dim, kernel_size=3, padding=1, stride=1),
            ConvModule(inter_dim, out_dim,   kernel_size=1, padding=0, stride=1),
        )

    def forward(self, x):
        h = self.res_layer(x)
        return x + h if self.shortcut else h

class ResStage(nn.Module):
    def __init__(self, in_dim, out_dim, num_blocks=1) -> None:
        super(ResStage, self).__init__()
        # ----------------- Network setting -----------------
        res_blocks = []
        for i in range(num_blocks):
            if i == 0:
                shortcut = (in_dim == out_dim)
                res_blocks.append(ResBlock(in_dim, out_dim, shortcut))
            else:
                res_blocks.append(ResBlock(out_dim, out_dim, True))
        self.res_blocks = nn.Sequential(*res_blocks)

    def forward(self, x):
        return self.res_blocks(x)


# ------------ VQ-GAN Encoder ------------
class VqGanEncoder(nn.Module):
    def __init__(self, img_dim=3, hidden_dim=256):
        super().__init__()
        # 4x downsampling
        self.layer_1 = ConvModule(img_dim, hidden_dim // 2, kernel_size=4, padding=1, stride=2)
        self.layer_2 = ConvModule(hidden_dim // 2, hidden_dim, kernel_size=4, padding=1, stride=2)

        self.layer_3 = ResStage(hidden_dim, hidden_dim, num_blocks=2)
        self.layer_4 = ResStage(hidden_dim, hidden_dim, num_blocks=2)
        
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
        
        return x

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
    
class VqGanDecoder(nn.Module):
    def __init__(self, img_dim=3, hidden_dim=256, latent_dim=128):
        super().__init__()
        # 2x upsampling
        self.layer_1 = ResStage(latent_dim, hidden_dim, num_blocks=3)
        self.layer_2 = DeConvModule(hidden_dim, hidden_dim // 2, kernel_size=4, padding=1, stride=2)
        # 4x upsampling
        self.layer_3 = ResStage(hidden_dim // 2, hidden_dim // 2, num_blocks=3)
        self.layer_4 = DeConvModule(hidden_dim // 2, hidden_dim // 4, kernel_size=4, padding=1, stride=2)
        self.last_conv = nn.Conv2d(hidden_dim // 4, img_dim, kernel_size=3, padding=1, stride=1)

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

        x = self.last_conv(x)

        return x


# ------------------ VQ-VAE ------------------
class VQGAN(nn.Module):
    def __init__(self, img_dim=3, num_embeddings = 512, hidden_dim=128, latent_dim=64):
        super().__init__()
        self.img_dim = img_dim
        self.hidden_dim = hidden_dim
        self.num_embeddings = num_embeddings
        self.latent_dim = latent_dim
        
        self.encoder  = VqGanEncoder(img_dim, hidden_dim)
        self.decoder  = VqGanDecoder(img_dim, hidden_dim, latent_dim)
        self.codebook = CodeBook(num_embeddings, hidden_dim, latent_dim)
    
    def compute_loss(self, x, x_rec, vq_output):
        # ----------- Reconstruction loss -----------
        rec_loss = F.l1_loss(x, x_rec, reduction='mean')

        # ----------- Latent loss -----------
        emb_loss = vq_output['emb_loss']

        # Total loss
        loss_dict = {
            'rec_loss': rec_loss,
            'emb_loss': emb_loss,
        }

        return loss_dict
    
    def forward_encode(self, x):
        # Encode
        z_e = self.encoder(x)

        # Quantize
        vq_output = self.codebook(z_e)
        z_q = vq_output['rep_z_q']
        bs, c, h, w = z_q.shape

        # Token ids
        ids = vq_output['min_encodings']
        ids = ids.view(bs, h*w,)     # [BHW,] -> [B, HW,]

        return z_q, ids
    
    def forward_decode(self, z_q):
        x_rec = self.decoder(z_q)
        return x_rec
    
    def forward(self, x):
        # Encode
        z_e = self.encoder(x)

        # Quantize
        vq_output = self.codebook(z_e)
        z_q = vq_output['rep_z_q']

        # Decode
        x_rec = self.decoder(z_q)

        output = {
            'x': x,
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

    print(' \n=========== VAE Encoder =========== ')
    # Prepare an image as the input
    bs, img_dim, img_size = 2, 3, 128
    hidden_dim = 128
    latent_dim = 64
    x = torch.randn(bs, img_dim, img_size, img_size)

    # Build model
    model = VqGanEncoder(img_dim=img_dim, hidden_dim=hidden_dim)

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
    model = VqGanDecoder(img_dim=img_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)

    # Inference
    outputs = model(x)
    print(outputs.shape)

    # Compute FLOPs & Params
    model.eval()
    x = torch.randn(1, latent_dim, img_size // 4, img_size // 4)
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('Decoder FLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Decoder Params : {:.2f} M'.format(params / 1e6))


    print(' \n=========== VQ-VAE =========== ')
    # Prepare an image as the input
    bs, img_dim, img_size = 2, 3, 128
    x = torch.randn(bs, img_dim, img_size, img_size)

    # Build model
    model = VQGAN(img_dim=img_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, num_embeddings=512)

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