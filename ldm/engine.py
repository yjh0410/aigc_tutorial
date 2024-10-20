import lpips
import torch
import torch.nn.functional as F

from typing import Iterable

import torch.utils

from utils.misc import MetricLogger, SmoothedValue
from utils.misc import print_rank_0
from utils.lr_scheduler import LinearWarmUpLrScheduler

from discriminator import PatchDiscriminator
from metric import batch_calculate_psnr, batch_calculate_ssim


def calculate_lambda(last_layer, p_loss, g_loss):
    p_loss_grads = torch.autograd.grad(p_loss, last_layer, retain_graph=True)[0]
    g_loss_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

    λ = torch.norm(p_loss_grads) / (torch.norm(g_loss_grads) + 1e-4)
    λ = torch.clamp(λ, 0, 1e4).detach().contiguous()

    return 0.8 * λ

def first_stage_train_one_epoch(args,
                                device              :torch.device,
                                model               :torch.nn.Module,
                                discriminator       :PatchDiscriminator,
                                lpips_loss_f        :lpips.LPIPS,
                                data_loader         :Iterable,
                                optimizer_G         :torch.optim.AdamW,
                                optimizer_D         :torch.optim.AdamW,
                                lr_scheduler_warmup :LinearWarmUpLrScheduler,
                                epoch               :int,
                                local_rank          :int = 0,
                                ):
    model.train()
    discriminator.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr_G',    SmoothedValue(window_size=1, fmt='{value:.7f}'))
    metric_logger.add_meter('lr_D',    SmoothedValue(window_size=1, fmt='{value:.7f}'))
    metric_logger.add_meter('gnorm_G', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('gnorm_D', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{} / {}]'.format(epoch, args.max_epoch)
    print_freq = 20
    epoch_size = len(data_loader)

    optimizer_G.zero_grad()
    optimizer_D.zero_grad()

    disc_start = epoch_size * 5

    # Train one epoch
    for iter_i, (images, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        ni = iter_i + epoch * epoch_size
        disc_factor = 0.0 if ni < disc_start else 1.0

        # Warmup
        if ni < args.wp_iters:
            lr_scheduler_warmup(ni, optimizer_G)
            lr_scheduler_warmup(ni, optimizer_D)
        elif ni == args.wp_iters:
            print("Warmup stage is over.")
            lr_scheduler_warmup.set_lr(optimizer_G, lr_scheduler_warmup.base_lr)
            lr_scheduler_warmup.set_lr(optimizer_D, lr_scheduler_warmup.base_lr)

        # To device
        images = images.to(device, non_blocking=True)   # original images: [B, C, H, W]
        labels = labels.to(device, non_blocking=True)   # class labels: [B,]

        # Inference
        output = model(images)

        # ------------- Discriminator loss -------------
        for p in discriminator.parameters():
            p.requires_grad = True
        
        disc_loss = discriminator.calc_disc_loss(real_x=images, fake_x=output["x_pred"])
        disc_loss = disc_factor * disc_loss

        # Backward
        optimizer_D.zero_grad()
        disc_loss.backward()
        gnorm_D = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=4.0)

        # Optimize
        optimizer_D.step()

        # ------------- Generator loss -------------
        for p in discriminator.parameters():
            p.requires_grad = False

        # Adversarial loss
        adv_loss = discriminator.calc_adv_loss(fake_x=output["x_pred"])

        # L1 pixel loss
        rec_loss = F.l1_loss(output["x_pred"], images, reduction="mean")

        # LPIPS loss
        real_x_norm = images * 2.0 - 1.0
        fake_x_norm = output["x_pred"] * 2.0 - 1.0
        lpips_loss = lpips_loss_f(real_x_norm, fake_x_norm)
        lpips_loss = torch.mean(lpips_loss)

        # Calculate loss weight for GAN loss
        rec_lpips_loss = 1.0 * lpips_loss + 1.0 * rec_loss
        if args.distributed:
            λ = calculate_lambda(model.module.decoder.layers[-1].weight, rec_lpips_loss, adv_loss)
        else:
            λ = calculate_lambda(model.decoder.layers[-1].weight, rec_lpips_loss, adv_loss)

        # KL loss
        kl_loss  = torch.mean(0.5 * (-1.0 + output["log_var"].exp() + torch.square(output["mu"]) - output["log_var"]))

        # Generator loss
        gen_loss = rec_lpips_loss + 0.00001 * kl_loss + λ * disc_factor * adv_loss

        # Backward
        optimizer_G.zero_grad()
        gen_loss.backward()
        gnorm_G = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)

        # Optimize
        optimizer_G.step()

        # ------------- Output log info. -------------
        # Logs
        lr_G = optimizer_G.param_groups[0]["lr"]
        lr_D = optimizer_D.param_groups[0]["lr"]
        nloss_dict = {}
        nloss_dict['l1_loss']    = rec_loss
        nloss_dict['lpips_loss'] = lpips_loss
        nloss_dict['kl_loss']    = kl_loss
        nloss_dict['adv_loss']   = adv_loss
        nloss_dict['disc_loss']  = disc_loss
        metric_logger.update(**nloss_dict)
        metric_logger.update(lr_G=lr_G)
        metric_logger.update(lr_D=lr_D)
        metric_logger.update(gnorm_G=gnorm_G)
        metric_logger.update(gnorm_D=gnorm_D)

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
