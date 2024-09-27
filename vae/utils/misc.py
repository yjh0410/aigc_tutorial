import time
import numpy as np
import random
import datetime
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist

from .distributed_utils import get_world_size, is_main_process, is_dist_avail_and_initialized


# ---------------------- Common functions ----------------------
def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x

def print_rank_0(msg, rank=None):
    if rank is not None and rank <= 0:
        print(msg)
    elif is_main_process():
        print(msg)

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


# ---------------------- Optimize functions ----------------------
def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device)
                                        for p in parameters]),
                            norm_type)

    return total_norm

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward()
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


# ---------------------- Model functions ----------------------
def load_model(args, model_without_ddp, optimizer, lr_scheduler):
    args.start_epoch = 0
    if args.resume and args.resume.lower() != 'none':
        print("=================== Load checkpoint ===================")
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            print('- Load optimizer from the checkpoint. ')
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1

        if 'lr_scheduler' in checkpoint:
            print('- Load lr scheduler from the checkpoint. ')
            lr_scheduler.load_state_dict(checkpoint.pop("lr_scheduler"))

def save_model(args, model_without_ddp, optimizer, lr_scheduler, epoch, metrics=None):
    output_dir = Path(args.output_dir)
    to_save = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'args': args,
    }
    if metrics is None:
        checkpoint_path = output_dir / ('checkpoint-{}.pth'.format(epoch))
    else:
        checkpoint_path = output_dir / ('checkpoint-{}_psnr_{}_ssim_{}.pth'.format(epoch, metrics[0], metrics[1]))
        to_save["psnr"] = metrics[0]
        to_save["ssim"] = metrics[1]

    torch.save(to_save, checkpoint_path)

def save_sampler(args, model_without_ddp, optimizer, lr_scheduler, epoch):
    output_dir = Path(args.output_dir)
    to_save = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'args': args,
    }
    checkpoint_path = output_dir / ('checkpoint-{}.pth'.format(epoch))        
    torch.save(to_save, checkpoint_path)


# ---------------------- Tools for GAN ----------------------
def load_gan_model(args, generator, lpips, discriminator, optimizer_G, optimizer_D, lr_scheduler_G, lr_scheduler_D):
    args.start_epoch = 0
    if args.resume and args.resume.lower() != 'none':
        print("=================== Load checkpoint ===================")
        checkpoint = torch.load(args.resume, map_location='cpu')
        generator.load_state_dict(checkpoint['generator'])
        print("Resume checkpoint for Generator: %s" % args.resume)
        
        lpips.load_state_dict(checkpoint['lpips'])
        print("Resume checkpoint for LPIPS: %s" % args.resume)
        
        discriminator.load_state_dict(checkpoint['discriminator'])
        print("Resume checkpoint for Discriminator: %s" % args.resume)
        
        # Optimizer for Generator
        if 'optimizer_G' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            print('- Load optimizer from the checkpoint. ')
            optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            args.start_epoch = checkpoint['epoch'] + 1

        # Optimizer for Discriminator
        if 'optimizer_D' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            print('- Load optimizer from the checkpoint. ')
            optimizer_D.load_state_dict(checkpoint['optimizer_D'])
            args.start_epoch = checkpoint['epoch'] + 1

        # LrScheduler for Generator
        if 'lr_scheduler_G' in checkpoint:
            print('- Load lr scheduler from the checkpoint. ')
            lr_scheduler_G.load_state_dict(checkpoint.pop("lr_scheduler_G"))

        # LrScheduler for Discriminator
        if 'lr_scheduler_D' in checkpoint:
            print('- Load lr scheduler from the checkpoint. ')
            lr_scheduler_D.load_state_dict(checkpoint.pop("lr_scheduler_D"))

def save_gan_model(args, generator, lpips, discriminator, optimizer_G, optimizer_D, lr_scheduler_G, lr_scheduler_D, epoch, metrics=None):
    output_dir = Path(args.output_dir)
    to_save = {
        'generator':      generator.state_dict(),
        'lpips':          lpips.state_dict(),
        'discriminator':  discriminator.state_dict(),
        'optimizer_G':    optimizer_G.state_dict(),
        'optimizer_D':    optimizer_D.state_dict(),
        'lr_scheduler_G': lr_scheduler_G.state_dict(),
        'lr_scheduler_D': lr_scheduler_D.state_dict(),
        'epoch': epoch,
        'args': args,
    }
    if metrics is None:
        checkpoint_path = output_dir / ('checkpoint_{}.pth'.format(epoch))
    else:
        checkpoint_path = output_dir / ('checkpoint_{}_psnr_{}_ssim_{}.pth'.format(epoch, metrics[0], metrics[1]))
        to_save['psnr'] = metrics[0]
        to_save['ssim'] = metrics[1]

    torch.save(to_save, checkpoint_path)
