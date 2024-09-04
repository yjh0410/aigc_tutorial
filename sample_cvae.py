import cv2
import argparse
import numpy as np

# ---------------- Torch compoments ----------------
import torch
import torch.backends.cudnn as cudnn

# ---------------- Model compoments ----------------
from models import build_model

# ---------------- Utils compoments ----------------
from utils.misc import setup_seed


def parse_args():
    parser = argparse.ArgumentParser()
    # Basic
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed.')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda')
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset name')
    parser.add_argument('--root', type=str, default='/mnt/share/ssd2/dataset',
                        help='path to dataset folder')
    # Model
    parser.add_argument('--model', type=str, default='vae',
                        help='model name')
    parser.add_argument('--weight', default=None, type=str,
                        help='keep training')

    return parser.parse_args()

    
def main():
    args = parse_args()
    # set random seed
    # setup_seed(args.seed)

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

    # ------------------------- Data information -------------------------
    if args.dataset == 'mnist':
        args.img_dim = 1
        args.latent_size = [7, 7]

    if args.dataset == 'cifar10':
        args.img_dim = 3
        args.latent_size = [8, 8]

    # ------------------------- Build Model -------------------------
    model = build_model(args)
    checkpoint = torch.load(args.weight, map_location='cpu')
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # ----------------- Sample for CVAE -----------------
    with torch.no_grad():   
        num_samples = 100
        for j in range(1, num_samples):
            for i in range(10):
                zi = torch.randn(1, model.latent_dim, *args.latent_size).to(device)
                label = torch.ones([zi.shape[0], 1, zi.shape[2], zi.shape[3]]).to(zi.device) \
                        * i
                print('Condition label: ', label)
                czi = torch.cat([zi, label], dim=1)
                rec_x = model.forward_decode(czi)
                
                rec_img = rec_x[0].permute(1, 2, 0).cpu().numpy()
                rec_img = np.uint8(np.clip(rec_img * 255, 0, 255))

                cv2.imshow("reconstructed image", rec_img)
                cv2.waitKey(0)

if __name__ == "__main__":
    main()