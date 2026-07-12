import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


class RiemannianCayleySGD(Optimizer):

    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(RiemannianCayleySGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                G = p.grad.data
                V = p.data

                if getattr(p, 'is_stiefel', False):
                    D, R = V.size()

                    U = torch.cat([G, V], dim=-1)
                    Y = torch.cat([V, -G], dim=-1)

                    YTU = torch.matmul(Y.T, U)
                    I = torch.eye(2*R, device=V.device, dtype=V.dtype)

                    M = I+(lr/2)*YTU
                    M_inv = torch.linalg.inv(M)

                    YTV = torch.matmul(Y.T, V)
                    step_dir = torch.matmul(U, torch.matmul(M_inv, YTV))

                    V_new = V-lr*step_dir
                    p.data.copy_(V_new)
                else:
                    p.data.add_(G, alpha=-lr)

        return loss
