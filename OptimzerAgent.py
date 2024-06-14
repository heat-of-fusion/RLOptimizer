import torch
import torch.nn as nn
from torch.optim import Optimizer
import torch.distributions as dist

import A2CNets as net

class OptimAgentDemoV2(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(OptimAgentDemoV2, self).__init__(params, defaults)

        self.beta_1 = betas[0]
        self.beta_2 = betas[1]
        self.gamma_3 = 1.0

        return

    def update_param(self, beta_1, beta_2, gamma_3):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.gamma_3 = gamma_3

        return

    def step(self):
        loss = None
        for group in self.param_groups:

            for p in group['params']:
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                b1, b2 = group['betas']
                b1 = self.beta_1
                b2 = self.beta_2

                # print(f'b1: {b1}, b2: {b2}')

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                exp_avg = torch.mul(exp_avg, b1) + (1 - b1) * grad

                exp_avg_sq = torch.mul(exp_avg_sq, b2) + (1 - b2) * (grad * grad)

                mhat = exp_avg / (1 - b1 ** state['step'])
                vhat = exp_avg_sq / (1 - b2 ** state['step'])

                denom = torch.sqrt(vhat + group['eps'])

                p.data = p.data - group['lr'] * mhat / denom * self.gamma_3

                state['exp_avg'], state['exp_avg_sq'] = exp_avg, exp_avg_sq

        return loss


class OptimAgentDemo(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(OptimAgentDemo, self).__init__(params, defaults)

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        self.gamma_3 = 1.0

        return

    def update_param(self, gamma_1, gamma_2, gamma_3):
        self.gamma_1 = gamma_1.cpu().detach().numpy()[0]
        self.gamma_2 = gamma_2.cpu().detach().numpy()[0]
        self.gamma_3 = gamma_3.cpu().detach().numpy()[0]

        # print(f'update: {[self.gamma_1, self.gamma_2, self.gamma_3]}')

        return

    def step(self):
        loss = None
        for group in self.param_groups:

            for p in group['params']:
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                b1, b2 = group['betas']
                b1 = b1 * self.gamma_1
                b2 = b2 * self.gamma_2

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                exp_avg = torch.mul(exp_avg, b1) + (1 - b1) * grad

                exp_avg_sq = torch.mul(exp_avg_sq, b2) + (1 - b2) * (grad * grad)

                mhat = exp_avg / (1 - b1 ** state['step'])
                vhat = exp_avg_sq / (1 - b2 ** state['step'])

                denom = torch.sqrt(vhat + group['eps'])

                p.data = p.data - group['lr'] * mhat / denom * self.gamma_3

                state['exp_avg'], state['exp_avg_sq'] = exp_avg, exp_avg_sq

        return loss

class OptimAgent():
    def __init__(self, state_size, hidden_size = 256, lr = 3e-4):
        self.a2c_net = net.A2CNet(state_size, hidden_size)
        self.optimizer = torch.optim.Adam(self.a2c_net.parameteres(), lr = lr)

        self.discount_factor = 0.99

    def get_action(self, state):
        (gamma_1_mu, gamma_1_sigma), (gamma_2_mu, gamma_2_sigma), (gamma_3_mu, gamma_3_sigma), _ = self.a2c_net(state)
        gamma_1_dist = dist.Normal(torch.tensor([gamma_1_mu]), torch.tensor([gamma_1_sigma]))
        gamma_2_dist = dist.Normal(torch.tensor([gamma_2_mu]), torch.tensor([gamma_2_sigma]))
        gamma_3_dist = dist.Normal(torch.tensor([gamma_3_mu]), torch.tensor([gamma_3_sigma]))

        gamma_1 = gamma_1_dist.sample((1, ))
        gamma_2 = gamma_2_dist.sample((1, ))
        gamma_3 = gamma_3_dist.sample((1, ))

        return gamma_1, gamma_2, gamma_3