import cv2
import argparse
import numpy as np

# ---------------- Torch compoments ----------------
import torch
import torch.backends.cudnn as cudnn

# ---------------- Model compoments ----------------
from dataset import build_dataset, build_dataloader
from models  import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    # Basic
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed.')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda')
    parser.add_argument('--sample', action='store_true', default=False,
                        help='sample or reconstruct.')
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset name')
    parser.add_argument('--root', type=str, default='/mnt/share/ssd2/dataset',
                        help='path to dataset folder')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers')
    # Model
    parser.add_argument('--model', type=str, default='vqvae',
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

    # ------------------------- Build Dataset -------------------------
    dataset    = build_dataset(args, is_train=False)
    dataloader = build_dataloader(args, dataset, is_train=False)

    # ------------------------- Build Model -------------------------
    model = build_model(args)
    checkpoint = torch.load(args.weight, map_location='cpu')
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()

    # ----------------- Sample for VAE -----------------
    with torch.no_grad():   
        if args.sample:
            # We randomly mask the given data as the prompt.
            pass
        else:
            # Do encoding with codebook and decoding to reconstruct
            for sample in dataloader:
                images = sample[0].to(device)
                output = model(images)
                x_rec  = output['x_pred']

                for bi in range(images.shape[0]):
                    # Predicted image
                    x_hat = x_rec[bi].permute(1, 2, 0).cpu().numpy() * 255.
                    x_hat = x_hat.astype(np.uint8)

                    # Original image
                    x_org = images[bi].permute(1, 2, 0).cpu().numpy() * 255.
                    x_org = x_org.astype(np.uint8)

                    x_vis = np.concatenate([x_org, x_hat], axis=1)
                    x_vis = cv2.cvtColor(x_vis, cv2.COLOR_RGB2BGR)

                    cv2.imshow("reconstructed image", x_vis)
                    cv2.waitKey(0)

if __name__ == "__main__":
    main()