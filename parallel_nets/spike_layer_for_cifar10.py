import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class MLF_unit(nn.Module):
    def __init__(self, vtr, m, theta_star, delta, Isyn_pre, Gsyn_pre, gleak):
        super(MLF_unit, self).__init__()
        self.vtr = vtr
        self.m = m
        self.theta_star = theta_star
        self.delta = delta
        self.Isyn_pre = Isyn_pre
        self.Gsyn_pre = Gsyn_pre
        self.gleak = gleak

    def forward(self, x):
        firetime = list()
        # for i in range(1, self.m):
        #     if self.vtr[i - 1] < self.theta_star:
        #         self.vtr[i] = self.vtr[i - 1] - self.delta * self.gleak * self.vtr[i - 1] - \
        #                       self.delta * np.sum(self.Gsyn_pre[i - 1]) * self.vtr[i - 1] + self.delta * np.sum(self.Isyn_pre[i - 1])
        #     else:
        #         firetime.append(i)
        return firetime


# class tdBatchNorm(nn.Module):
#     def __init__(self, bn, alpha=1):
#         super(tdBatchNorm, self).__init__()
#         self.bn = bn
#         self.alpha = alpha
#
#     def forward(self, x):
#         exponential_average_factor = 0.0
#
#         if self.training and self.bn.track_running_stats:
#             if self.bn.num_batches_tracked is not None:
#                 self.bn.num_batches_tracked += 1
#                 if self.bn.momentum is None:  # use cumulative moving average
#                     exponential_average_factor = 1.0 / float(self.bn.num_batches_tracked)
#                 else:  # use exponential moving average
#                     exponential_average_factor = self.bn.momentum
#
#         if self.training:
#             mean = x.mean([0, 2, 3], keepdim=True)
#             var = x.var([0, 2, 3], keepdim=True, unbiased=False)
#             n = x.numel() / x.size(1)
#             with torch.no_grad():
#                 self.bn.running_mean = exponential_average_factor * mean[0, :, 0, 0]\
#                                        + (1 - exponential_average_factor) * self.bn.running_mean
#                 self.bn.running_var = exponential_average_factor * var[0, :, 0, 0] * n / (n - 1) \
#                                       + (1 - exponential_average_factor) * self.bn.running_var
#         else:
#             mean = self.bn.running_mean[None, :, None, None]
#             var = self.bn.running_var[None, :, None, None]
#
#         x = self.alpha * Vth_p * (x - mean) / (torch.sqrt(var) + self.bn.eps)
#
#         if self.bn.affine:
#             x = x * self.bn.weight[None, :, None, None] + self.bn.bias[None, :, None, None]
#
#         return x

class SpikeFunction(torch.autograd.Function):
    @staticmethod
    # 在计算突触权重
    def forward(ctx, input):
        ctx.save_for_backward(input)

        return

    @staticmethod
    def backward(ctx, grad_output):
        # 更新突触权重
        input, = ctx.saved_tensors

        return

spikefunc = SpikeFunction.apply





