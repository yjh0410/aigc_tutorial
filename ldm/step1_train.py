import os
import time
import argparse
import datetime
import lpips

# ---------------- Torch compoments ----------------
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ---------------- Dataset compoments ----------------
from dataset import build_dataset, build_dataloader

# ---------------- Model compoments ----------------
from autoencoder import VariationalAE
from discriminator import PatchDiscriminator

# ---------------- Utils compoments ----------------
from utils import distributed_utils
from utils.misc import setup_seed, print_rank_0, load_gan_model, save_gan_model
from utils.lr_scheduler import LinearWarmUpLrScheduler

# ---------------- Training engine ----------------
from engine import first_stage_train_one_epoch
from engine import evaluate_one_epoch


def parse_args():
    parser = argparse.ArgumentParser()
    # Basic
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed.')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--output_dir', type=str, default='weights/',
                        help='path to save trained model.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate model.')
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset name')
    parser.add_argument('--root', type=str, default='/mnt/share/ssd2/dataset',
                        help='path to dataset folder')
    parser.add_argument('--img_dim', type=int, default=3,
                        help='input image channels.')
    parser.add_argument('--img_size', type=int, default=64,
                        help='input image size.')
    # Model
    parser.add_argument('--model', type=str, default='first_stage_vae',
                        help='model name')
    parser.add_argument('--resume', default=None, type=str,
                        help='keep training')
    # Optimizer
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='training optimier.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='gradient accumulation')
    parser.add_argument('--lr', type=float, default=0.00002,
                        help='initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='initial learning rate.')
    parser.add_argument('--wp_iters', type=float, default=500,
                        help='initial learning rate.')
    # Epoch
    parser.add_argument('--max_epoch', type=int, default=20,
                        help='number of workers')
    parser.add_argument('--eval_epoch', type=int, default=5,
                        help='number of workers')
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
    args.output_dir = os.path.join(args.output_dir, args.dataset, args.model)
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    valid_dataset = build_dataset(args, is_train=False)
    valid_dataloader = build_dataloader(args, valid_dataset, is_train=False)

    print(' ============= Dataset Info. ============= ')
    print("Dataset: ", args.dataset)
    print('- train dataset size : ', len(train_dataset))

    # ------------------------- Build VQ-GAN -------------------------
    print(' ============= VAE Info. ============= ')
    model = VariationalAE(img_dim=args.img_dim, hidden_dims=[32, 64, 128, 256], latent_dim=4)
    model = model.to(device)
    print(model)

    # ------------------------- Build Discriminator -------------------------
    print(' ============= Discriminator Info. ============= ')
    pdisc = PatchDiscriminator(in_dim=3, ndf=64, n_layers=3)
    pdisc = pdisc.to(device)
    print(pdisc)

    # ------------------------- Build LPIPS -------------------------
    print(' - Use LPIPS to train VAE in first stage. ')
    lpips_loss = lpips.LPIPS(net="vgg").eval()
    lpips_loss = lpips_loss.to(device)

    # ------------------------- Build Warmup LR Scheduler -------------------------
    lr_scheduler_wp = LinearWarmUpLrScheduler(args.lr, wp_iters=args.wp_iters)

    # ------------------------- Build Optimzier & Scheduler for Generator -------------------------
    optimizer_G = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    lr_scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=args.max_epoch - 1, eta_min=0.01)

    # ------------------------- Build Optimzier & Scheduler for Discriminator -------------------------
    optimizer_D = torch.optim.AdamW(pdisc.parameters(), lr=args.lr, weight_decay=0.0)
    lr_scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=args.max_epoch - 1, eta_min=0.01)

    # ------------------------- Build Criterion -------------------------
    load_gan_model(args, model, pdisc, optimizer_G, optimizer_D, lr_scheduler_G, lr_scheduler_D)

    # ------------------------- Build DDP Model -------------------------
    model_wo_ddp = model
    pdisc_wo_ddp = pdisc
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_wo_ddp = model.module

        pdisc = DDP(pdisc, device_ids=[args.gpu])
        pdisc_wo_ddp = pdisc.module

    # ------------------------- Evaluate first -------------------------
    if args.eval:
        metrics = evaluate_one_epoch(device, model_wo_ddp, valid_dataloader, local_rank=0)
        print("PSNR: ", round(metrics["psnr"], 4))
        print("SSIM: ", round(metrics["ssim"], 4))
        return

    # ------------------------- Training Pipeline -------------------------
    start_time = time.time()
    print_rank_0("=============== Step-1: Train AutoEncoder ===============", local_rank)
    for epoch in range(args.start_epoch, args.max_epoch):
        if args.distributed:
            train_dataloader.batch_sampler.sampler.set_epoch(epoch)

        # Train one epoch
        first_stage_train_one_epoch(args,
                                    device,
                                    model,
                                    pdisc,
                                    lpips_loss,
                                    train_dataloader,
                                    optimizer_G,
                                    optimizer_D,
                                    lr_scheduler_wp,
                                    epoch,
                                    local_rank,
                                    tblogger,
                                    )
        
        # LR scheduler
        lr_scheduler_G.step()
        lr_scheduler_D.step()

        # Evaluate
        if local_rank <= 0:
            if (epoch % args.eval_epoch) == 0 or (epoch + 1 == args.max_epoch):
                # Save model
                metrics = evaluate_one_epoch(device, model_wo_ddp, valid_dataloader, local_rank=0)
                print('- saving the model after {} epochs ...'.format(epoch))

                # Save the checkpoint
                psnr, ssim = round(metrics["psnr"], 2), round(metrics["ssim"], 4)
                save_gan_model(args, model_wo_ddp, pdisc_wo_ddp,
                               optimizer_G, optimizer_D,
                               lr_scheduler_G, lr_scheduler_D,
                               epoch, metrics=[psnr, ssim])
        if args.distributed:
            dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    main()
