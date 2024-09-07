import torch
from typing import Iterable

from utils.misc import MetricLogger, SmoothedValue
from utils.misc import print_rank_0

from metric import batch_calculate_psnr, batch_calculate_ssim


def train_one_epoch(args,
                    device,
                    model,
                    data_loader,
                    optimizer,
                    lr_scheduler_warmup,
                    epoch,
                    local_rank=0,
                    tblogger=None,
                    ):
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{} / {}]'.format(epoch, args.max_epoch)
    print_freq = 20
    epoch_size = len(data_loader)

    optimizer.zero_grad()

    # train one epoch
    for iter_i, (images, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        ni = iter_i + epoch * epoch_size
        nw = lr_scheduler_warmup.wp_iters

        # Warmup
        if   ni <  nw:
            lr_scheduler_warmup(ni, optimizer)
        elif ni == nw:
            print("Warmup stage is over.")
            lr_scheduler_warmup.set_lr(optimizer, lr_scheduler_warmup.base_lr)

        # To device
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Inference
        output = model(images)

        # Training losses
        loss_dict = output['loss_dict']
        losses = loss_dict['loss']

        # Backward
        losses.backward()

        # Optimize
        optimizer.step()
        optimizer.zero_grad()

        # Logs
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(**loss_dict)
        metric_logger.update(lr=lr)

        if tblogger is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((iter_i / len(data_loader) + epoch) * 1000)
            for k in loss_dict:
                tblogger.add_scalar(k, loss_dict[k].item(), epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print_rank_0("Averaged stats: {}".format(metric_logger), local_rank)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_sampler_one_epoch(args,
                            device,
                            sampler,
                            data_loader,
                            optimizer,
                            lr_scheduler_warmup,
                            epoch,
                            local_rank=0,
                            tblogger=None,
                            ):
    sampler.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{} / {}]'.format(epoch, args.max_epoch)
    print_freq = 20
    epoch_size = len(data_loader)

    optimizer.zero_grad()

    # train one epoch
    for iter_i, (images, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        ni = iter_i + epoch * epoch_size
        nw = args.wp_iters

        # Warmup
        if   ni <  nw:
            lr_scheduler_warmup(ni, optimizer)
        elif ni == nw:
            print("Warmup stage is over.")
            lr_scheduler_warmup.set_lr(optimizer, lr_scheduler_warmup.base_lr)

        # To device
        images = images.to(device, non_blocking=True)   # original images: [B, C, H, W]
        labels = labels.to(device, non_blocking=True)   # class labels: [B,]

        # Inference
        output = sampler(images)

        # Training losses
        loss_dict = output['loss_dict']
        losses = loss_dict['loss']

        # Backward
        losses.backward()

        # Optimize
        optimizer.step()
        optimizer.zero_grad()

        # Logs
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(**loss_dict)
        metric_logger.update(lr=lr)

        if tblogger is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((iter_i / len(data_loader) + epoch) * 1000)
            for k in loss_dict:
                tblogger.add_scalar(k, loss_dict[k].item(), epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print_rank_0("Averaged stats: {}".format(metric_logger), local_rank)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate_one_epoch(device     : torch.device,
                       model      : torch.nn.Module,
                       dataloader : Iterable,
                       local_rank : int,
                       ):
    # Set eval mode
    model.eval()

    # Initialize the Metric Logger
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 10
    epoch_size = len(dataloader)

    # Train one epoch pipeline
    for iter_i, (images, labels) in enumerate(metric_logger.log_every(dataloader, print_freq, header)):
        # To device
        images = images.to(device, non_blocking=True)   # original images: [B, C, H, W]
        labels = labels.to(device, non_blocking=True)   # class labels: [B,]

        # Inference
        outputs = model(images)
        x_rec  = outputs['x_pred']

        # Calculate metrics
        psnr = batch_calculate_psnr(x_rec, images)
        ssim = batch_calculate_ssim(x_rec, images)

        # Update log
        batch_size = x_rec.shape[0]
        metric_logger.meters['psnr'].update(psnr.item(), n=batch_size)
        metric_logger.meters['ssim'].update(ssim.item(), n=batch_size)

    # gather the stats from all processes
    print_rank_0('* PSNR {psnr.global_avg:.3f} SSIM {ssim.global_avg:.3f}'
                 .format(psnr=metric_logger.psnr, ssim=metric_logger.ssim),
                 local_rank)
    metrics = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return metrics
