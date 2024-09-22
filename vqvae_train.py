import os
import time
import argparse
import datetime

# ---------------- Torch compoments ----------------
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ---------------- Dataset compoments ----------------
from dataset import build_dataset, build_dataloader

# ---------------- Model compoments ----------------
from models.vqvae import VQVAE

# ---------------- Utils compoments ----------------
from utils import distributed_utils
from utils.misc import setup_seed, print_rank_0, load_model, save_model
from utils.lr_scheduler import LinearWarmUpLrScheduler

# ---------------- Training engine ----------------
from vqvae_engine import train_one_epoch, evaluate_one_epoch


def parse_args():
    parser = argparse.ArgumentParser()
    # Basic
    parser.add_argument('--batch_size', type=int, default=128,
                        help='gradient accumulation')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed.')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--path_to_save', type=str, default='weights/',
                        help='path to save trained model.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate model.')
    # Epoch
    parser.add_argument('--max_epoch', type=int, default=20,
                        help='number of workers')
    parser.add_argument('--eval_epoch', type=int, default=5,
                        help='number of workers')
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset name')
    parser.add_argument('--root', type=str, default='/mnt/share/ssd2/dataset',
                        help='path to dataset folder')
    parser.add_argument('--img_dim', type=int, default=3,
                        help='number of workers')
    parser.add_argument('--img_size', type=int, default=64,
                        help='number of workers')
    # Model
    parser.add_argument('--model', type=str, default='vae',
                        help='model name')
    parser.add_argument('--resume', default=None, type=str,
                        help='keep training')
    # Optimizer
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='training optimier.')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='initial learning rate.')
    parser.add_argument('--wp_iters', type=float, default=500,
                        help='initial learning rate.')
    # DDP
    parser.add_argument('--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='the number of local rank.')

    return parser.parse_args()

    
def main():
    args = parse_args()
    # set random seed
    setup_seed(args.seed)

    # Path to save model
    path_to_save = os.path.join(args.path_to_save, args.dataset, args.model)
    os.makedirs(path_to_save, exist_ok=True)
    args.output_dir = path_to_save
    
    # ------------------------- Build DDP environment -------------------------
    ## LOCAL_RANK is the global GPU number tag, the value range is [0, world_size - 1].
    ## LOCAL_PROCESS_RANK is the number of the GPU of each machine, not global.
    local_rank = local_process_rank = -1
    if args.distributed:
        distributed_utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(distributed_utils.get_sha()))
        try:
            # Multiple Mechine & Multiple GPUs (world size > 8)
            local_rank = torch.distributed.get_rank()
            local_process_rank = int(os.getenv('LOCAL_PROCESS_RANK', '0'))
        except:
            # Single Mechine & Multiple GPUs (world size <= 8)
            local_rank = local_process_rank = torch.distributed.get_rank()
    print_rank_0(args)
    args.world_size = distributed_utils.get_world_size()
    print('World size: {}'.format(distributed_utils.get_world_size()))
    print("LOCAL RANK: ", local_rank)
    print("LOCAL_PROCESS_RANL: ", local_process_rank)

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

    # ------------------------- Build Tensorboard -------------------------
    tblogger = None
    if local_rank <= 0 and args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        time_stamp = time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))
        log_path = os.path.join('log/', args.dataset, args.model, time_stamp)
        os.makedirs(log_path, exist_ok=True)
        tblogger = SummaryWriter(log_path)

    # ------------------------- Build Dataset -------------------------
    train_dataset = build_dataset(args, is_train=True)
    train_dataloader = build_dataloader(args, train_dataset, is_train=True)
    tvalid_dataset = build_dataset(args, is_train=False)
    valid_dataloader = build_dataloader(args, tvalid_dataset, is_train=False)

    print('=================== Dataset Information ===================')
    print("Dataset: ", args.dataset)
    print('- train dataset size : ', len(train_dataset))

    # ------------------------- Build Model -------------------------
    model = VQVAE(args.img_dim, num_embeddings=512, hidden_dim=128, latent_dim=64)
    model.train().to(device)
    print(model)

    # ------------------------- Build Optimzier & Scheduler -------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch - 1, eta_min=0.0)
    lr_scheduler_wp = LinearWarmUpLrScheduler(args.lr, wp_iters=args.wp_iters)

    # ------------------------- Build DDP Model -------------------------
    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # ------------------------- Build Criterion -------------------------
    load_model(args, model_without_ddp, optimizer, lr_scheduler)

    # ------------------------- Evaluate first -------------------------
    if args.eval:
        metrics = evaluate_one_epoch(device, model_without_ddp, valid_dataloader, local_rank=0)
        print("PSNR: ", round(metrics["psnr"], 4))
        print("SSIM: ", round(metrics["ssim"], 4))
        return

    # ------------------------- Training Pipeline -------------------------
    start_time = time.time()
    print_rank_0("=============== Step-1: Train VQ-VAE  ===============", local_rank)
    for epoch in range(args.start_epoch, args.max_epoch):
        if args.distributed:
            train_dataloader.batch_sampler.sampler.set_epoch(epoch)

        # train one epoch
        train_one_epoch(args,
                        device,
                        model,
                        train_dataloader,
                        optimizer,
                        lr_scheduler_wp,
                        epoch,
                        local_rank,
                        tblogger,
                        )

        # LR scheduler
        lr_scheduler.step()

        # Evaluate
        if local_rank <= 0:
            if (epoch % args.eval_epoch) == 0 or (epoch + 1 == args.max_epoch):
                metrics = evaluate_one_epoch(device, model_without_ddp, valid_dataloader, local_rank=0)

                # Save model
                print('- saving the model after {} epochs ...'.format(epoch))
                psnr, ssim = round(metrics["psnr"], 2), round(metrics["ssim"], 4)
                save_model(args, model_without_ddp, optimizer, lr_scheduler, epoch, metrics=[psnr, ssim])

        # Waiting for the main process
        if args.distributed:
            dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    main()