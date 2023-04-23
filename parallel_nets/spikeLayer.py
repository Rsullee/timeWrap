import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

TimeStep = 4

def _compute_exp(m, exp_temp, spikeTime, taus):
    for i in range(1, m):
        exp_temp[i] = (exp_temp[i - 1] + spikeTime[i - 1]) * np.exp(-1 / taus)


def _vmax_subthr_noresponse(vmax_subthr_all, vtr, theta_star):
    if vtr[0] > vtr[1]:
        vmax_subthr_all[0] = vtr[0]
    for i in range(1, len(vtr) - 1):
        if vtr[i] > vtr[i + 1] and vtr[i] > vtr[i - 1] and vtr[i] < theta_star:
            vmax_subthr_all[i] = vtr[i]
    if vtr[len(vtr) - 1] > vtr[len(vtr) - 2]:
        vmax_subthr_all[len(vtr) - 1] = vtr[len(vtr) - 1]
    return vmax_subthr_all


def _computevtr(vtr, m, theta_star, delta, Isyn_pre, Gsyn_pre, gleak):
    firetime = list()
    for i in range(1, m):
        if vtr[i - 1] < theta_star:
            # vtr[i] = vtr[i-1] - delta * vtr[i-1] * (gleak + np.sum(Gsyn_pre[i-1])) + delta * np.sum(Isyn_pre[i-1])
            vtr[i] = vtr[i - 1] - delta * gleak * vtr[i - 1] - delta * np.sum(Gsyn_pre[i - 1]) * vtr[
                i - 1] + delta * np.sum(Isyn_pre[i - 1])
        else:
            firetime.append(i)
        # vtr[i] = vtr[i-1] - delta * gleak * vtr[i-1] - delta * np.sum(Gsyn_pre[i-1]) * vtr[i-1] + delta * np.sum(Isyn_pre[i-1])
        # if vtr[i] >= theta_star :
        #     vtr[i] = 0
        #     firetime.append(i)
        # temp_G.append(sum(Gsyn_pre[i]))
    return firetime


class tempotronGMulti(nn.Module):
    def __init__(self, t, n, dt, gleak, taus, beta, mu, threshold, learning_rate, ex, inh, w_mean, sp, test=False,
                 flag=-1):
        super(tempotronGMulti, self).__init__()
        self.T = t
        self.N = n
        self.dt = dt
        self.gleak = gleak
        self.taus = taus
        self.beta = beta
        self.mu = mu
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.Vrev = self.init_vrev(ex, inh)
        self.ex = ex
        self.inh = inh
        self.w = self.init_weights(w_mean, n)
        self.spike = sp
        self.test = test
        self.flag = flag
        self.w_pre = np.zeros(self.n)
        self.midVariables()

    def forward(self, vtr):

        return firetime


    def midVariables(self):
        # self.m = int(round(self.T / self.delta)) # 这个有点问题 core dumped
        self.m = int(self.T / self.delta)
        self.GList = np.zeros(self.m)

        self.firetime = list()

        # 发放的脉冲个数初始为0
        self.output = 0
        # 存储所有的local峰值
        self.vmax_subthr_all = np.zeros(int(self.T / self.delta), dtype=np.float64)
        # 存储所有峰值的位置
        self.vmaxi_subthr_all = np.zeros(int(self.T / self.delta), dtype=np.intc)
        # subthreshold vmax
        self.vmax_subthr = 0
        # 记录vmax的时间
        self.vmaxi_subthr = 0
        # 每个时刻的膜电位,指定了dtype数据类型
        self.vtr = np.zeros(self.m, dtype=np.float64)
        # previous vtr, 观察用的
        self.vtr_pre = np.zeros(self.m, dtype=np.float64)
        # 根据coba的公式，求的每个时刻对于wi的偏导,PartialDerivativeTrace
        self.PDTrace = np.zeros((self.m, self.n), dtype=np.float64)
        # 这个值不理解
        # self.vmax_thr = 0.01
        # delta_w,元素为delta_wi,数量为突触的个数
        self.dw = np.zeros(self.n, dtype=np.float64)
        # previous dw
        self.dw_pre = np.zeros(self.n, dtype=np.float64)
        # previous Isyn
        # previous Gsyn
        self.Isyn_pre = np.zeros((self.m, self.n), dtype=np.float64)
        self.Gsyn_pre = np.zeros((self.m, self.n), dtype=np.float64)
        # self.exp = np.zeros(len(self.spikeTime))
        self.exp = np.zeros(self.n, dtype=np.float64)
        # previous exp
        self.exp_temp = np.zeros((self.m, self.n), dtype=np.float64)

        self.tLTP = 0
        self.tLTD = 0
        self.VLTD = 0

    def time2spike(self, sp):
        spike_times = sp['spikeTimes'].flatten() * self.beta
        synapse_ids = sp['synapseIds'].flatten()

        spike_times = spike_times * 1000  # 向上取整
        spike_times = ((1 / self.delta) * spike_times).astype(int)

        spikes = np.zeros((math.ceil(self.T / self.delta), self.n))  # 向上取整

        # 去掉这个循环
        for i in range(0, len(spike_times)):
            spikes[spike_times[i]][synapse_ids[i]] = 1

        self.midVariables()
        return spikes

    def init_weights(self, w_mean=0.01, n=500):
        synapseWeights_init = np.random.randn(n)  # 0为均值、以1为标准差的正态分布，记为N（0，1）
        synapseWeights_mean = w_mean
        synapseWeights_std = 0.01
        synapseWeights_init = synapseWeights_mean + synapseWeights_std * synapseWeights_init

        return synapseWeights_init

    def init_vrev(self, ex=5, inh=-1):
        reversalPotential = np.zeros(self.n)
        bili = 5
        reversalPotential[0: int(len(reversalPotential) / 10) * bili] = ex
        reversalPotential[int(len(reversalPotential) / 10) * (10 - bili):] = inh
        np.random.shuffle(reversalPotential)

        return reversalPotential

    def response_noinit(self):
        self.output = 0
        # 因为重新计算了synapseWeight， 所以需要重新计算
        self.compute_vtr()
        self.output = np.sum(self.vtr >= self.theta_star)
        return self.output


    def compute_vtr(self):
        self.reversal()
        self.vtr.fill(0)
        self.exp_temp.fill(0)
        self.Isyn_pre.fill(0)
        self.Gsyn_pre.fill(0)
        self.Isyn()
        self.Gsyn()

        self.firetime = list()
        self.firetime = _computevtr(self.vtr, self.m, self.theta_star, self.delta, self.Isyn_pre, self.Gsyn_pre,
                                    self.gleak)

    def Isyn(self):
        self.compute_exp()
        self.Isyn_pre = self.Vrev * np.abs(self.w) * self.exp_temp
        # self.Isyn_pre = self.Vrev * self.w * self.exp_temp
        # self.Isyn_pre = _Isyn(self.Vrev, self.w, self.exp_temp)

    def Gsyn(self):
        self.Gsyn_pre = np.abs(self.w) * self.exp_temp
        # self.Gsyn_pre = self.w * self.exp_temp
        # self.Gsyn_pre = _Gsyn(self.w, self.exp_temp)

    def compute_exp(self):
        self.exp_temp.fill(0)
        _compute_exp(self.m, self.exp_temp, self.spikes, self.taus)

    # 导数只计算到需要修改的时刻就可以
    def compute_PDTrace(self, time):

        self.PDTrace.fill(0)
        # times = np.arange(0,self.T,self.delta)
        # for index in range(1, time + 1):
        #     # # self.exp其实计算的是某一个时刻的，那么这里再用self.exp就不对了
        #     u_1 = self.PDTrace[index - 1] * (1 - self.delta * self.gleak  - self.delta * np.sum(self.Gsyn_pre[index - 1]))
        #     v_1 = -1 * self.vtr[index-1] * self.delta * self.exp_temp[index-1]
        #     w_1 = self.delta * self.Vrev * self.exp_temp[index-1]
        #     self.PDTrace[index] =  u_1 + w_1 + v_1

        for index in range(1, time + 1):
            tmpExp = self.exp_temp[index - 1]
            tmpIndex = np.where(self.w < 0)
            tmpExp[tmpIndex] = -1 * tmpExp[tmpIndex]

            u_1 = self.PDTrace[index - 1] * (
                    1 - self.delta * self.gleak - self.delta * np.sum(self.Gsyn_pre[index - 1]))
            v_1 = -1 * self.vtr[index - 1] * self.delta * tmpExp
            w_1 = self.delta * self.Vrev * tmpExp
            self.PDTrace[index] = u_1 + w_1 + v_1

    def vmax_subthr_noresponse(self):

        # 找这个位置时要考虑好多情况
        # 1. vtr曲线的形状有哪些
        self.vmax_subthr_all.fill(float('-inf'))
        self.vmaxi_subthr_all.fill(0)

        self.vmax_subthr_all = _vmax_subthr_noresponse(self.vmax_subthr_all, self.vtr, self.theta_star)

        self.vmax_subthr = max(self.vmax_subthr_all)
        self.vmaxi_subthr = self.vmax_subthr_all.argmax()  # 原先的

        # self.vmaxi_subthr = np.where(self.vmax_subthr_all!=float('-inf'))[0][-1]   # 最后一个
        # if self.vmax_subthr==min(self.vmax_subthr_all) :    # 不太合理
        #     self.vmaxi_subthr = int(self.m/2)
        epoch = 0
        # 如果subthr的位置为0或1，
        while self.vmaxi_subthr == 0 or self.vmaxi_subthr == 1 or self.vmaxi_subthr == self.m - 1:
            tmp = self.vmax_subthr
            self.vmax_subthr_all[self.vmaxi_subthr] = min(self.vmax_subthr_all)
            tmpi = self.vmaxi_subthr
            self.vmaxi_subthr = self.vmax_subthr_all.argmax()
            self.vmax_subthr_all[tmpi] = tmp
            if epoch > 3:
                self.vmaxi_subthr = int(self.m / 2)  # 这个地方不合理
            # if epoch > 10:
            #     for i in range(len(self.vmax_subthr_all)):
            #         if self.vmax_subthr_all[i] != float('-inf'):
            #             self.vmaxi_subthr = i
            #             break
            #     break
            epoch += 1
            # if epoch > 20:
            #     for i in range(len(self.vmax_subthr_all)):
            #         if self.vmax_subthr_all[i] != float('-inf'):
            #             self.vmaxi_subthr = i
            #             break

        def train_notheta_noresponse(self, lable):

            # self.dw.fill(0)
            spike_error = lable - self.output
            if spike_error == 0:
                return 0
            elif spike_error < 0:
                self.compute_dw_notheta_ltd_noresponse()
            else:
                self.compute_dw_notheta_ltp_noresponse()

        # 发放少了
        def compute_dw_notheta_ltp_noresponse(self):

            self.vmax_subthr_noresponse()
            # 一个整数序号
            self.tLTP = self.vmaxi_subthr

            self.compute_PDTrace(self.tLTP)
            self.dw = self.learningRate * self.PDTrace[self.tLTP] + self.mu * self.dw
            # self.dw = self.learningRate * self.PDTrace[tLTP]
            # self.w_pre = self.w
            self.w += self.dw

        # 发放多了
        def compute_dw_notheta_ltd_noresponse(self):

            # 因为计算output时计算了一边，没有必要再计算一遍
            # self.compute_vtr()
            v_over_theta = self.vtr[self.vtr >= self.theta_star]
            self.VLTD = v_over_theta.min()
            self.tLTD = np.where(self.vtr == self.VLTD)[0][0]

            # v_over_theta = self.vtr[self.vtr >= self.theta_star]      # 修改最后一个脉冲
            # self.VLTD = v_over_theta[-1]
            # self.tLTD = np.where(self.vtr == self.VLTD)[0][0]

            if self.tLTD == 0:  # 如果是脉冲在第一个位置，那么将这个位置变成最大的，然后再找最小的就不会是第一个了
                v_over_theta[v_over_theta == self.VLTD] = max(v_over_theta)
                self.VLTD = v_over_theta.min()
                self.tLTD = np.where(self.vtr == self.VLTD)[0][0]

            self.compute_PDTrace(self.tLTD)
            self.dw = -self.learningRate * self.PDTrace[self.tLTD] + self.mu * self.dw
            self.w_pre = self.w
            self.w += self.dw

class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, ctx, vtr):
        ctx.save_for_backward(vtr, self)
        firetime = list()
        if vtr < self.theta_star:
            vtr = vtr - self.delta * self.gleak * vtr - self.delta * np.sum(self.Gsyn_pre) * vtr + self.delta * np.sum(self.Isyn_pre)
        # else:
        #     firetime.append(i)
        return vtr

    @staticmethod
    def backward(self, ctx, grad_out):
        exp_temp = (self.exp_temp + self.spikeTime) * np.exp(-1 / self.taus)
        tmpExp = self.exp_temp
        tmpIndex = np.where(self.w < 0)
        tmpExp[tmpIndex] = -1 * tmpExp[tmpIndex]

        u_1 = self.PDTrace[index - 1] * (1 - self.delta * self.gleak - self.delta * np.sum(self.Gsyn_pre[index - 1]))
        v_1 = -1 * self.vtr[index - 1] * self.delta * tmpExp
        w_1 = self.delta * self.Vrev * tmpExp
        self.PDTrace[index] = u_1 + w_1 + v_1
        return self.PDTrace

spikefunc = SpikeFunction.apply