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
from models import build_model, build_sampler

# ---------------- Utils compoments ----------------
from utils import distributed_utils
from utils.misc import setup_seed, print_rank_0, load_model, save_sampler
from utils.lr_scheduler import LinearWarmUpLrScheduler

# ---------------- Training engine ----------------
from vqvae_engine import train_sampler_one_epoch


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
    parser.add_argument('--output_dir', type=str, default='weights/',
                        help='path to save trained model.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate model.')
    # Epoch
    parser.add_argument('--max_epoch', type=int, default=100,
                        help='number of workers')
    parser.add_argument('--eval_epoch', type=int, default=5,
                        help='number of workers')
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset name')
    parser.add_argument('--root', type=str, default='/mnt/share/ssd2/dataset',
                        help='path to dataset folder')
    # Model
    parser.add_argument('--model', type=str, default='vqvae',
                        help='model name')
    parser.add_argument('--sampler', default="gpt_small", type=str,
                        help='transformer sampler')
    parser.add_argument('--resume', default=None, type=str,
                        help='keep training')
    parser.add_argument('--vqvae_checkpoint', default=None, type=str,
                        help='Load trained well VAE model from the given checkpoint')
    # Optimizer
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.01,
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
    args.output_dir = os.path.join(args.output_dir, args.dataset, args.model, args.sampler)
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

    print('=================== Dataset Information ===================')
    print("Dataset: ", args.dataset)
    print('- train dataset size : ', len(train_dataset))

# ------------------------- Build Model & Sampler -------------------------
    print(" =================== VAE Model info. =================== ")
    vqvae_model = build_model(args)
    if args.vqvae_checkpoint is not None:
        print(f' - Load checkpoint for VQ-VAE from the checkpoint : {args.vqvae_checkpoint} ...')
        vqvae_model.load_state_dict(torch.load(args.vqvae_checkpoint, map_location="cpu").pop("model"))
    vqvae_model = vqvae_model.eval().to(device)
    print(vqvae_model)

    print(" =================== VAE Sampler info. =================== ")
    sampler = build_sampler(args, vqvae_model.num_embeddings)
    sampler = sampler.train().to(device)
    print(sampler)

# ------------------------- Build Optimzier & Scheduler -------------------------
    decay, no_decay = set(), set()
    for mn, m in sampler.transformer.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn

            if pn.endswith("bias"):
                no_decay.add(fpn)

            elif pn.endswith("weight") and isinstance(m, torch.nn.Linear):
                decay.add(fpn)

            elif pn.endswith("weight") and isinstance(m, torch.nn.Embedding):
                no_decay.add(fpn)

    param_dict = {pn: p for pn, p in sampler.transformer.named_parameters()}
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95))

    # ------------------------- Build Optimzier & Scheduler -------------------------
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch - 1, eta_min=0.0)
    lr_scheduler_wp = LinearWarmUpLrScheduler(args.lr, wp_iters=args.wp_iters)

    # ------------------------- Build DDP Model -------------------------
    model_without_ddp = sampler
    if args.distributed:
        model = DDP(sampler, device_ids=[args.gpu])
        model_without_ddp = model.module

    # ------------------------- Build Criterion -------------------------
    load_model(args, model_without_ddp, optimizer, lr_scheduler)

    # ------------------------- Training Pipeline -------------------------
    start_time = time.time()
    print_rank_0("=============== Step-2: Train Sampler  ===============", local_rank)
    for epoch in range(args.start_epoch, args.max_epoch):
        if args.distributed:
            train_dataloader.batch_sampler.sampler.set_epoch(epoch)

        # train one epoch
        train_sampler_one_epoch(args,
                                device,
                                vqvae_model,
                                sampler,
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
                # Save model
                print('- saving the model after {} epochs ...'.format(epoch))
                save_sampler(args, model_without_ddp, optimizer, lr_scheduler, epoch)

        # Waiting for the main process
        if args.distributed:
            dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    main()