import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class FSM(nn.Module):

    def __init__(self, p=0.2, alpha=0.1, eps=1e-06, mix='crossdomain'):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps
        self.mix = mix
        self._activated = True

    def set_activation(self, activated):
        self._activated = activated

    def set_probability(self, p):
        self.p = p

    def forward(self, x, domain_labels=None):
        if not self.training or not self._activated:
            return x
        if random.random() > self.p:
            return x
        B, C, H, W = x.shape
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        x_normed = (x - mu) / sig
        if self.mix == 'crossdomain' and domain_labels is not None:
            perm = self._get_crossdomain_perm(domain_labels, x.device)
        else:
            perm = torch.randperm(B).to(x.device)
        mu2 = mu[perm]
        sig2 = sig[perm]
        lmda = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)).to(x.device)
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)
        return x_normed * sig_mix + mu_mix

    def _get_crossdomain_perm(self, domain_labels, device):
        B = len(domain_labels)
        perm = torch.zeros(B, dtype=torch.long, device=device)
        domain_labels = domain_labels.to(device)
        for i in range(B):
            current_domain = domain_labels[i].item()
            if current_domain == 0:
                candidates = ((domain_labels == 1) | (domain_labels == 2)).nonzero(as_tuple=True)[0]
            else:
                candidates = (domain_labels == 0).nonzero(as_tuple=True)[0]
            if len(candidates) > 0:
                perm[i] = candidates[random.randint(0, len(candidates) - 1)]
            else:
                perm[i] = random.randint(0, B - 1)
        return perm

class CrossNorm(nn.Module):

    def __init__(self, p=0.2, eps=1e-06):
        super().__init__()
        self.p = p
        self.eps = eps
        self._activated = True

    def set_activation(self, activated):
        self._activated = activated

    def set_probability(self, p):
        self.p = p

    def forward(self, x, domain_labels=None):
        if not self.training or not self._activated:
            return x
        if random.random() > self.p:
            return x
        B, C, H, W = x.shape
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        x_normed = (x - mu) / sig
        if domain_labels is not None:
            perm = self._get_crossdomain_perm(domain_labels, x.device)
        else:
            perm = torch.randperm(B).to(x.device)
        mu_swap = mu[perm]
        sig_swap = sig[perm]
        return x_normed * sig_swap + mu_swap

    def _get_crossdomain_perm(self, domain_labels, device):
        B = len(domain_labels)
        perm = torch.zeros(B, dtype=torch.long, device=device)
        domain_labels = domain_labels.to(device)
        for i in range(B):
            current_domain = domain_labels[i].item()
            if current_domain == 0:
                candidates = ((domain_labels == 1) | (domain_labels == 2)).nonzero(as_tuple=True)[0]
            else:
                candidates = (domain_labels == 0).nonzero(as_tuple=True)[0]
            if len(candidates) > 0:
                perm[i] = candidates[random.randint(0, len(candidates) - 1)]
            else:
                perm[i] = random.randint(0, B - 1)
        return perm

class AdaptiveFSM(nn.Module):

    def __init__(self, p_start=0.3, p_end=0.1, alpha=0.1, eps=1e-06, mix='crossdomain'):
        super().__init__()
        self.p_start = p_start
        self.p_end = p_end
        self.alpha = alpha
        self.eps = eps
        self.mix = mix
        self._current_p = p_start
        self._activated = True
        self.fsm = FSM(p=p_start, alpha=alpha, eps=eps, mix=mix)

    def set_progress(self, progress):
        if progress < 0.3:
            self._current_p = self.p_start
        else:
            decay_progress = (progress - 0.3) / 0.7
            self._current_p = self.p_start - (self.p_start - self.p_end) * decay_progress
        self.fsm.set_probability(self._current_p)

    def set_activation(self, activated):
        self._activated = activated
        self.fsm.set_activation(activated)

    def forward(self, x, domain_labels=None):
        if not self._activated:
            return x
        return self.fsm(x, domain_labels)

    @property
    def current_p(self):
        return self._current_p

class StyleAugmentor(nn.Module):

    def __init__(self, method='fsm', p=0.2, alpha=0.1, adaptive=True, p_start=0.3, p_end=0.1):
        super().__init__()
        self.method = method
        self.adaptive = adaptive
        if adaptive:
            self.fsm = AdaptiveFSM(p_start, p_end, alpha, mix='crossdomain')
            self.crossnorm = None
        else:
            self.fsm = FSM(p=p, alpha=alpha, mix='crossdomain')
            self.crossnorm = CrossNorm(p=p)

    def set_progress(self, progress):
        if self.adaptive and hasattr(self.fsm, 'set_progress'):
            self.fsm.set_progress(progress)

    def forward(self, x, domain_labels=None):
        if self.method == 'fsm':
            return self.fsm(x, domain_labels)
        elif self.method == 'crossnorm':
            return self.crossnorm(x, domain_labels)
        elif self.method == 'both':
            x = self.fsm(x, domain_labels)
            if self.crossnorm is not None:
                x = self.crossnorm(x, domain_labels)
            return x
        return x
if __name__ == '__main__':
    print('测试FSM模块...')
    B = 8
    x = torch.randn(B, 256, 56, 56)
    domain_labels = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])
    fsm = FSM(p=1.0, alpha=0.1, mix='crossdomain')
    fsm.train()
    out = fsm(x, domain_labels)
    print(f'FSM输出形状: {out.shape}')
    crossnorm = CrossNorm(p=1.0)
    crossnorm.train()
    out = crossnorm(x, domain_labels)
    print(f'CrossNorm输出形状: {out.shape}')
    adaptive = AdaptiveFSM(p_start=0.3, p_end=0.1)
    adaptive.train()
    for progress in [0.0, 0.3, 0.5, 0.8, 1.0]:
        adaptive.set_progress(progress)
        print(f'Progress={progress:.1f}, p={adaptive.current_p:.3f}')
    print('\n所有测试通过!')
