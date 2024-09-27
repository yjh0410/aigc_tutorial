from utils.misc import MetricLogger, SmoothedValue
from utils.misc import print_rank_0


def train_one_epoch(args,
                        device,
                        model,
                        data_loader,
                        optimizer,
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
    for iter_i, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        ni = iter_i + epoch * epoch_size

        # To device
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

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
