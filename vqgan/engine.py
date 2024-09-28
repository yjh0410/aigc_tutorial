import torch
import torch.nn.functional as F

from typing import Iterable

from utils.misc import MetricLogger, SmoothedValue
from utils.misc import print_rank_0

from metric import batch_calculate_psnr, batch_calculate_ssim


def calculate_lambda(last_layer, p_loss, g_loss):
    p_loss_grads = torch.autograd.grad(p_loss, last_layer, retain_graph=True)[0]
    g_loss_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

    λ = torch.norm(p_loss_grads) / (torch.norm(g_loss_grads) + 1e-4)
    λ = torch.clamp(λ, 0, 1e4).detach().contiguous()

    return 0.8 * λ

def train_one_epoch(args,
                    device,
                    model,
                    lpips,
                    pdisc,
                    data_loader,
                    optimizer_G,
                    optimizer_D,
                    lr_scheduler_warmup,
                    epoch,
                    local_rank=0,
                    tblogger=None,
                    ):
    model.train()
    pdisc.train()
    lpips.eval()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr_G', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('lr_D', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{} / {}]'.format(epoch, args.max_epoch)
    print_freq = 20
    epoch_size = len(data_loader)

    optimizer_G.zero_grad()
    optimizer_D.zero_grad()

    disc_start = epoch_size * 3

    # Train one epoch
    for iter_i, (images, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        ni = iter_i + epoch * epoch_size
        nw = args.wp_iters

        # Warmup
        if   ni <  nw:
            lr_scheduler_warmup(ni, optimizer_G)
            lr_scheduler_warmup(ni, optimizer_D)
        elif ni == nw:
            print("Warmup stage is over.")
            lr_scheduler_warmup.set_lr(optimizer_G, lr_scheduler_warmup.base_lr)
            lr_scheduler_warmup.set_lr(optimizer_D, lr_scheduler_warmup.base_lr)

        # To device
        images = images.to(device, non_blocking=True)   # original images: [B, C, H, W]
        labels = labels.to(device, non_blocking=True)   # class labels: [B,]

        # Inference
        output = model(images)
        loss_dict = output['loss_dict']  # include reconstruction loss

        # ------------- Generator loss -------------
        # We wish the generator to fool the discriminator,
        # so the goal is to make the output as close to 1 as possible.
        real_x = output['x']
        fake_x = output['x_pred']
        disc_fake = pdisc(fake_x)
        g_loss    = -disc_fake.mean()

        ploss = lpips(real_x, fake_x)
        rloss = loss_dict['rec_loss']
        qloss = loss_dict['emb_loss']
        pr_loss = 1.0 * ploss + 1.0 * rloss

        # Calculate loss weight for GAN loss
        disc_factor = 0.0 if ni < disc_start else 1.0
        if args.distributed:
            λ = calculate_lambda(model.module.decoder.last_conv.weight, pr_loss, g_loss)
        else:
            λ = calculate_lambda(model.decoder.last_conv.weight, pr_loss, g_loss)

        # Generator loss
        vq_loss = ploss + rloss + qloss + λ * disc_factor * g_loss

        # ------------- Discriminator loss -------------
        # Discriminate real images
        disc_real = pdisc(real_x.detach())
        d_loss_real = torch.mean(F.relu(1. - disc_real))

        # Discriminate fake images
        disc_fake = pdisc(fake_x.detach())
        d_loss_fake = torch.mean(F.relu(1. + disc_fake))

        # Discriminator loss
        d_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

        # ------------- Backward & Optimize -------------
        # Backward Generator losses
        optimizer_G.zero_grad()
        vq_loss.backward(retain_graph=True)
        vq_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Backward Discriminator losses
        optimizer_D.zero_grad()
        d_loss.backward()

        torch.nn.utils.clip_grad_norm_(pdisc.parameters(), max_norm=1.0)

        # Optimize
        optimizer_G.step()
        optimizer_D.step()

        # ------------- Output log info. -------------
        # Logs
        lr_G = optimizer_G.param_groups[0]["lr"]
        lr_D = optimizer_D.param_groups[0]["lr"]
        nloss_dict = {}
        nloss_dict['rec_loss'] = rloss
        nloss_dict['emb_loss'] = qloss
        nloss_dict['ploss']    = ploss
        nloss_dict['d_loss']   = d_loss
        nloss_dict['g_loss']   = g_loss
        nloss_dict['d_loss_real'] = d_loss_real
        nloss_dict['d_loss_fake'] = d_loss_fake
        metric_logger.update(**nloss_dict)
        metric_logger.update(lr_G=lr_G)
        metric_logger.update(lr_D=lr_D)

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
                            vqvae_model,
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

        # Get token ids
        with torch.no_grad():
            z_q, tok_ids = vqvae_model.forward_encode(images)
        
        # Inference
        output = sampler(tok_ids.detach())

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
