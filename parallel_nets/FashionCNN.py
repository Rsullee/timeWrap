import torch
import torch.nn as nn
from neu_dynamics import mem_init, mem_update
import torch.nn.functional as F

class FashionCNN(nn.Module):
    def __init__(self, params):
        super(FashionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False)
        self.fc1 = nn.Linear(4 * 4 * 64, 1024, bias=False)
        self.fc2 = nn.Linear(1024, 10, bias=False)
        self.device = params['device']
        self.encoding = params['encoding']
        self.thresh = params['thresh']
        self.inf_mem = torch.tensor(params['inf_mem']).float().to(self.device)

    def forward(self, input, time_window=10):

        if self.encoding == 'latency':
            input_time = ((time_window - 1) - (time_window - 1) * input).round().long()
            input_spike = torch.zeros(time_window, *input_time.size(), device=self.device)
            input_spike = input_spike.scatter_(0, input_time.view(1, *input_time.size()), 1)

        for step in range(time_window):
            if self.encoding == 'real':
                input_spike = input
            elif self.encoding == 'poisson':
                input_spike = (input > torch.rand(input.size(), device=self.device)).float()
            elif self.encoding == 'latency':
                input_spike = input_spike[step]
            else:
                raise Exception('No valid code is specified.')

            if step == 0:
                c1_mem, c1_spike = mem_init(self.conv1(input_spike))

                p1_mem, p1_spike = mem_init(F.avg_pool2d(c1_spike, 2), True)

                c2_mem, c2_spike = mem_init(self.conv2(p1_spike))

                p2_mem, p2_spike = mem_init(F.avg_pool2d(c2_spike, 2), True)

                x = p2_spike.view(p2_spike.size(0), -1)

                h1_mem, h1_spike = mem_init(self.fc1(x))
                h2_mem, h2_spike = mem_init(self.fc2(h1_spike))
                output = torch.zeros(h2_spike.size(), device=self.device)
                max_mem = torch.where(h2_mem.detach() < self.thresh, h2_mem, -self.inf_mem)
                min_mem = torch.where(h2_spike.detach() > 0, h2_mem, self.inf_mem)
                output += h2_spike

            else:

                c1_mem, c1_spike = mem_update(self.conv1(input_spike), c1_mem)

                p1_mem, p1_spike = mem_update(F.avg_pool2d(c1_spike, 2), p1_mem, True)

                c2_mem, c2_spike = mem_update(self.conv2(p1_spike), c2_mem)

                p2_mem, p2_spike = mem_update(F.avg_pool2d(c2_spike, 2), p2_mem, True)

                x = p2_spike.view(p2_spike.size(0), -1)

                h1_mem, h1_spike = mem_update(self.fc1(x), h1_mem)
                h2_mem, h2_spike = mem_update(self.fc2(h1_spike), h2_mem)
                max_mem = torch.max(max_mem,
                                    torch.where(h2_mem.detach() < self.thresh, h2_mem, -self.inf_mem))
                min_mem = torch.min(min_mem,
                                    torch.where(h2_spike.detach() > 0, h2_mem, self.inf_mem))
                output += h2_spike

        return output, max_mem, min_mem
