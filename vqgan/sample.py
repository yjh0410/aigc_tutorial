import os
import cv2
import argparse
import numpy as np

# ---------------- Torch compoments ----------------
import torch
import torch.backends.cudnn as cudnn

# ---------------- Model compoments ----------------
from dataset import build_dataset, build_dataloader
from metric  import batch_calculate_psnr, batch_calculate_ssim

from model   import VQGAN
from sampler import build_gpt_sampler


def parse_args():
    parser = argparse.ArgumentParser()
    # Basic
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed.')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda')
    parser.add_argument('--task', type=str, default='reconstruction',
                        help='sample or reconstruct.')
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset name')
    parser.add_argument('--root', type=str, default='/mnt/share/ssd2/dataset',
                        help='path to dataset folder')
    parser.add_argument('--img_dim', type=int, default=3,
                        help='number of workers')
    parser.add_argument('--img_size', type=int, default=64,
                        help='number of workers')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers')
    # Model
    parser.add_argument('--model', type=str, default='vqgan',
                        help='model name')
    parser.add_argument('--sampler', default="gpt_small", type=str,
                        help='transformer sampler')
    parser.add_argument('--weight_vae', default=None, type=str,
                        help='Load the checkpoint of the vae model.')
    parser.add_argument('--weight_sampler', default=None, type=str,
                        help='Load the checkpoint of the vae sampler.')

    return parser.parse_args()


@torch.no_grad
def reconstruction(args, device):
    # Build Dataset
    dataset    = build_dataset(args, is_train=False)
    dataloader = build_dataloader(args, dataset, is_train=False)

    # Build Model
    # vqgan = VQGAN(args.img_dim, num_embeddings=512, hidden_dim=128, latent_dim=64)
    vqgan = VQGAN(img_dim=args.img_dim, num_embeddings=1024, hidden_dim=256, latent_dim=256)
    if args.weight_vae is not None:
        print(f' - Load checkpoint for VQ-GAN from the checkpoint : {args.weight_vae} ...')
        checkpoint = torch.load(args.weight_vae, map_location='cpu')
        vqgan.load_state_dict(checkpoint["model"])
    vqgan = vqgan.to(device).eval()

    # Output path
    output_dir = os.path.join("result", args.dataset, args.model)
    output_dir_org = os.path.join(output_dir, 'org')
    output_dir_rec = os.path.join(output_dir, 'rec')
    os.makedirs(output_dir_org, exist_ok=True)
    os.makedirs(output_dir_rec, exist_ok=True)

    # ----------------- Sample for VAE -----------------
    img_id = 0
    for sample in dataloader:
        images = sample[0].to(device)
        output = vqgan(images)
        x_rec  = output['x_pred']

        for bi in range(images.shape[0]):
            # Calculate metrics
            psnr = batch_calculate_psnr(x_rec[bi:bi+1], images[bi:bi+1])
            ssim = batch_calculate_ssim(x_rec[bi:bi+1], images[bi:bi+1])
            
            # Predicted image
            x_hat = np.clip(x_rec[bi].permute(1, 2, 0).cpu().numpy() * 255., 0.0, 255.)
            x_hat = x_hat.astype(np.uint8)

            # Original image
            x_org = np.clip(images[bi].permute(1, 2, 0).cpu().numpy() * 255., 0.0, 255.)
            x_org = x_org.astype(np.uint8)

            x_org = cv2.cvtColor(x_org, cv2.COLOR_RGB2BGR)
            x_hat = cv2.cvtColor(x_hat, cv2.COLOR_RGB2BGR)
            x_vis = np.concatenate([x_org, x_hat], axis=1)

            print(f" ============ Image ID-[{img_id}] / [{len(dataloader) * images.shape[0]}] ============")
            print(f" - PSNR : {round(psnr.item(), 2)}")
            print(f" - SSIM : {round(ssim.item(), 4)}")

            cv2.imwrite(os.path.join(output_dir_org, f"gt_{img_id}.png"), x_org)
            cv2.imwrite(os.path.join(output_dir_rec, f"rec_{img_id}.png"), x_hat)

            cv2.imshow("orig & rec", x_vis)
            cv2.waitKey(0)

            img_id += 1

@torch.no_grad
def completion(args, device):
    # Build Dataset
    dataset    = build_dataset(args, is_train=False)
    dataloader = build_dataloader(args, dataset, is_train=False)

    # Build Model
    vqgan = VQGAN(args.img_dim, num_embeddings=512, hidden_dim=128, latent_dim=64)
    if args.weight_vae is not None:
        print(f' - Load checkpoint for VQ-GAN from the checkpoint : {args.weight_vae} ...')
        checkpoint = torch.load(args.weight_vae, map_location='cpu')
        vqgan.load_state_dict(checkpoint["model"])
    vqgan = vqgan.to(device).eval()

    # Build Sampler
    vqgan_sampler = build_gpt_sampler(args, vqgan.num_embeddings)
    if args.weight_sampler is not None:
        checkpoint = torch.load(args.weight_sampler, map_location='cpu')
        vqgan_sampler.load_state_dict(checkpoint["model"])
    vqgan_sampler = vqgan_sampler.to(device).eval()

    # Output path
    output_dir = os.path.join("result", args.dataset, args.model)
    output_dir_org = os.path.join(output_dir, 'org')
    output_dir_completion = os.path.join(output_dir, 'completion')
    os.makedirs(output_dir_org, exist_ok=True)
    os.makedirs(output_dir_completion, exist_ok=True)

    # ---------------- Reconstruction pipeline ----------------
    img_id = 0
    for sample in dataloader:
        images = sample[0].to(device)   # original images: [B, C, H, W]

        # Encode to latent space
        z_q, tok_ids = vqgan.forward_encode(images)
        bs, c, h, w = z_q.shape
        seq_len = tok_ids.shape[1]

        # Initial token ids
        init_tok_ids = tok_ids[:, :int(seq_len * 0.5)]
        init_seq_len = init_tok_ids.shape[1]
        num_steps = seq_len - init_seq_len

        # Set SOS token id as the condition
        sos_tokens = torch.ones(init_tok_ids.shape[0], 1) * vqgan_sampler.sos_token
        sos_tokens = sos_tokens.long().to(init_tok_ids.device)

        # Sample by NTP
        print(" - Sampling by next-token-prediction (NTP) paradigm ...")
        tok_ids = vqgan_sampler.sample(init_tok_ids, condition=sos_tokens, num_steps=num_steps)

        # Get embeddings
        sampled_z_q = vqgan.codebook.embedding(tok_ids.view(-1))
        sampled_z_q = sampled_z_q.reshape(bs, h, w, c).permute(0, 3, 1, 2)

        # Decode images from the embeddings
        x_recs = vqgan.forward_decode(sampled_z_q)

        for bi in range(images.shape[0]):
            # Predicted image
            x_hat = np.clip(x_recs[bi].permute(1, 2, 0).cpu().numpy() * 255., 0.0, 255.)
            x_hat = x_hat.astype(np.uint8)

            # Original image
            x_org = np.clip(images[bi].permute(1, 2, 0).cpu().numpy() * 255., 0.0, 255.)
            x_org = x_org.astype(np.uint8)

            x_org = cv2.cvtColor(x_org, cv2.COLOR_RGB2BGR)
            x_hat = cv2.cvtColor(x_hat, cv2.COLOR_RGB2BGR)
            x_vis = np.concatenate([x_org, x_hat], axis=1)

            print(f" =========== Image ID-[{img_id}] / [{len(dataloader) * images.shape[0]}] ============ ")
            cv2.imwrite(os.path.join(output_dir_org, f"gt_{img_id}.png"), x_org)
            cv2.imwrite(os.path.join(output_dir_completion, f"completion_{img_id}.png"), x_hat)

            cv2.imshow(f"original & reconstruct", x_vis)
            cv2.waitKey(0)

            img_id += 1

@torch.no_grad
def sample(args, device):
    args.img_dim = 3
    # Build Model
    vqgan = VQGAN(args.img_dim, num_embeddings=512, hidden_dim=128, latent_dim=64)
    if args.weight_vae is not None:
        print(f' - Load checkpoint for VQ-GAN from the checkpoint : {args.weight_vae} ...')
        checkpoint = torch.load(args.weight_vae, map_location='cpu')
        vqgan.load_state_dict(checkpoint["model"])
    vqgan = vqgan.to(device).eval()

    # Build Sampler
    vqgan_sampler = build_gpt_sampler(args, vqgan.num_embeddings)
    if args.weight_sampler is not None:
        checkpoint = torch.load(args.weight_sampler, map_location='cpu')
        vqgan_sampler.load_state_dict(checkpoint["model"])
    vqgan_sampler = vqgan_sampler.to(device).eval()

    # Output path
    output_dir = os.path.join("result", args.dataset, args.model, 'sample')
    os.makedirs(output_dir, exist_ok=True)

    # ---------------- Reconstruction pipeline ----------------
    img_id = 0
    for img_id in range(120):
        # 16 x 16 latent size for VQ-GAN train on the CelebA
        latent_h = 16
        latent_w = 16
        latent_dim = vqgan.latent_dim
        num_steps = latent_h * latent_w

        # Initial token ids
        init_tok_ids = torch.zeros([1, 0], dtype=torch.long).cuda()

        # Set SOS token id as the condition
        sos_tokens = torch.ones(init_tok_ids.shape[0], 1) * vqgan_sampler.sos_token
        sos_tokens = sos_tokens.long().to(init_tok_ids.device)

        # Sample by NTP
        print(" - Sampling by next-token-prediction (NTP) paradigm ...")
        tok_ids = vqgan_sampler.sample(init_tok_ids, condition=sos_tokens, num_steps=num_steps)

        # Get embeddings
        sampled_z_q = vqgan.codebook.embedding(tok_ids.view(-1))
        sampled_z_q = sampled_z_q.reshape(1, latent_h, latent_w, latent_dim).permute(0, 3, 1, 2)

        # Decode images from the embeddings
        x_sampled = vqgan.forward_decode(sampled_z_q)

        # Predicted image
        x_hat = np.clip(x_sampled[0].permute(1, 2, 0).cpu().numpy() * 255., 0.0, 255.)
        x_hat = x_hat.astype(np.uint8)

        x_hat = cv2.cvtColor(x_hat, cv2.COLOR_RGB2BGR)

        print(f" =========== Image ID-[{img_id}] ============ ")
        cv2.imwrite(os.path.join(output_dir, f"sampled_{img_id}.png"), x_hat)

        cv2.imshow(f"original & reconstruct", x_hat)
        cv2.waitKey(0)



def main():
    args = parse_args()

    # ------------------------- Build CUDA -------------------------
    if args.cuda:
        if torch.cuda.is_available():
            cudnn.benchmark = True
            device = torch.device("cuda")
        else:
            print('There is no available GPU.')
            args.cuda = False
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    # ------------ Run sampler ------------
    if args.task == "reconstruction":
        reconstruction(args, device)

    if args.task == "completion":
        completion(args, device)

    if args.task == "sample":
        sample(args, device)
        

if __name__ == "__main__":
    main()