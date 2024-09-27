import torch


# Basic Warmup Scheduler
class LinearWarmUpLrScheduler(object):
    def __init__(self, base_lr=0.01, wp_iters=500, warmup_factor=0.00066667):
        self.base_lr = base_lr
        self.wp_iters = wp_iters
        self.warmup_factor = warmup_factor

    def set_lr(self, optimizer, cur_lr):
        for param_group in optimizer.param_groups:
            init_lr = param_group['initial_lr']
            ratio = init_lr / self.base_lr
            param_group['lr'] = cur_lr * ratio

    def __call__(self, iter, optimizer):
        # warmup
        assert iter < self.wp_iters
        alpha = iter / self.wp_iters
        warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        tmp_lr = self.base_lr * warmup_factor
        self.set_lr(optimizer, tmp_lr)
