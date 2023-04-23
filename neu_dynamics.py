import torch
import numpy as np

thresh = None
thresh_pool = None
decay = None
delta = None
gleak = None
Isyn = None
Gsyn = None
exp = None
w = None
inh = None
ex = None
# vtr = None
spikes = None
m = None
t = None
Vrev = None
n = None


def param_init(threshold, threshold_pooling, decay_factor, deltaa, g_leak, Isyn_pre, Gsyn_pre, exp_temp, inhibit,
               excite, vtr_r, para_t, para_n, V_rev):
    global thresh, thresh_pool, decay, delta, gleak, Isyn, Gsyn, exp, w, inh, ex, spikes, m, t, Vrev, n
    thresh = threshold
    thresh_pool = threshold_pooling
    decay = decay_factor
    delta = deltaa
    gleak = g_leak
    Isyn = Isyn_pre
    Gsyn = Gsyn_pre
    exp = exp_temp

    inh = inhibit
    ex = excite
    w = init_weights(0.01, 500)

    # vtr = vtr_r

    t = para_t

    m = int(t / delta)
    n = para_n

    Vrev = V_rev

    Isyn = np.zeros((m, n), dtype=np.float64)
    Gsyn = np.zeros((m, n), dtype=np.float64)

def init_weights(w_mean=0.01, n=500):
    synapseWeights_init = np.random.randn(n)  # 0为均值、以1为标准差的正态分布，记为N（0，1）
    synapseWeights_mean = w_mean
    synapseWeights_std = 0.01
    synapseWeights_init = synapseWeights_mean + synapseWeights_std * synapseWeights_init
    return synapseWeights_init


class Box_Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vtr):
        # input: vtr
        ctx.save_for_backward(vtr)
        index = np.where(w < 0)
        Vrev[index] = inh
        index = np.where(w >= 0)
        Vrev[index] = ex

        # vtr.fill(0)
        exp.fill(0)

        Isyn = Vrev * np.abs(w) * exp
        Gsyn = np.abs(w) * exp
        # exp[i] = (exp_temp[i - 1] + spikeTime[i - 1]) * np.exp(-1 / taus)

        for i in range(1, m):
            if vtr[i - 1] < thresh:
                vtr[i] = vtr[i - 1] - delta * gleak * vtr[i - 1] - delta * np.sum(Gsyn[i - 1]) * vtr[
                    i - 1] + delta * np.sum(Isyn[i - 1])
        return vtr.ge(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        vtr, = ctx.saved_tensors
        grad_input = grad_output.clone()

        # temp = abs(input - thresh) < 0.5
        # return grad_input * temp.float()
        # update w
        # for index in range(1, time + 1):
        #     tmpExp = exp[index - 1]
        #     tmpIndex = np.where(w < 0)
        #     tmpExp[tmpIndex] = -1 * tmpExp[tmpIndex]
        #
        #     u_1 = PDTrace[index - 1] * (
        #                 1 - delta * gleak - delta * np.sum(Gsyn[index - 1]))
        #     v_1 = -1 * vtr[index - 1] * delta * tmpExp
        #     w_1 = delta * Vrev * tmpExp
        #     PDTrace[index] = u_1 + w_1 + v_1


class Box_Func_Pool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.ge(thresh_pool).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh_pool) < 0.5
        return grad_input * temp.float()


def mem_init(x, pooling=False):
    mem = x
    if not pooling:
        spike = Box_Func.apply(mem)
        mem = mem - spike * thresh
    else:
        spike = Box_Func_Pool.apply(mem)
        mem = mem - spike * thresh_pool
    return mem, spike


def mem_update(x, mem, pooling=False):
    mem = mem * decay + x
    if not pooling:
        spike = Box_Func.apply(mem)
        mem = mem - spike * thresh
    else:
        spike = Box_Func_Pool.apply(mem)
        mem = mem - spike * thresh_pool
    return mem, spike
