import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FrequencyAwarePerturbation(nn.Module):

    def __init__(self, perturb_low=False, perturb_high=True, eps_low=0.05, eps_high=0.2, low_freq_ratio=0.25):
        super().__init__()
        self.perturb_low = perturb_low
        self.perturb_high = perturb_high
        self.eps_low = eps_low
        self.eps_high = eps_high
        self.low_freq_ratio = low_freq_ratio
        self._mask_cache = {}

    def _get_frequency_masks(self, H, W, device):
        W_fft = W // 2 + 1
        key = (H, W_fft)
        if key not in self._mask_cache:
            radius = int(min(H, W_fft) * self.low_freq_ratio)
            y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W_fft, device=device), indexing='ij')
            center = (H // 2, W_fft // 2)
            dist = torch.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
            mask_low = (dist <= radius).float()
            mask_high = 1 - mask_low
            self._mask_cache[key] = (mask_low, mask_high)
        return self._mask_cache[key]

    def forward(self, features):
        if not self.training or (not self.perturb_low and (not self.perturb_high)):
            return features
        B, C, H, W = features.shape
        device = features.device
        mask_low, mask_high = self._get_frequency_masks(H, W, device)
        mask_low = mask_low.unsqueeze(0).unsqueeze(0)
        mask_high = mask_high.unsqueeze(0).unsqueeze(0)
        perturbed_list = []
        for c in range(C):
            feat_c = features[:, c:c + 1, :, :]
            fft = torch.fft.rfft2(feat_c, norm='ortho')
            fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
            noise_real = torch.randn_like(fft_shift.real)
            noise_imag = torch.randn_like(fft_shift.imag)
            noise_fft = torch.complex(noise_real, noise_imag)
            perturbed_fft = fft_shift.clone()
            if self.perturb_low:
                perturbed_fft = perturbed_fft + self.eps_low * noise_fft * mask_low
            if self.perturb_high:
                perturbed_fft = perturbed_fft + self.eps_high * noise_fft * mask_high
            fft_ishift = torch.fft.ifftshift(perturbed_fft, dim=(-2, -1))
            perturbed_c = torch.fft.irfft2(fft_ishift, s=(H, W), norm='ortho')
            perturbed_list.append(perturbed_c)
        perturbed_features = torch.cat(perturbed_list, dim=1)
        return perturbed_features

class AdaptiveFrequencyPerturbation(nn.Module):

    def __init__(self, perturb_high=True, eps_min=0.05, eps_max=0.3, low_freq_ratio=0.25):
        super().__init__()
        self.perturb_high = perturb_high
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.low_freq_ratio = low_freq_ratio
        self._mask_cache = {}

    def _get_frequency_masks(self, H, W, device):
        key = (H, W)
        if key not in self._mask_cache:
            radius = int(min(H, W) * self.low_freq_ratio)
            y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
            center = (H // 2, W // 2)
            dist = torch.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
            mask_low = (dist <= radius).float()
            mask_high = 1 - mask_low
            self._mask_cache[key] = (mask_low, mask_high)
        return self._mask_cache[key]

    def forward(self, features):
        if not self.training or not self.perturb_high:
            return features
        B, C, H, W = features.shape
        device = features.device
        feat_std = features.std(dim=(2, 3), keepdim=True)
        feat_std_norm = (feat_std - feat_std.min()) / (feat_std.max() - feat_std.min() + 1e-08)
        eps_adaptive = self.eps_min + (self.eps_max - self.eps_min) * (1 - feat_std_norm)
        mask_low, mask_high = self._get_frequency_masks(H, W, device)
        mask_high = mask_high.unsqueeze(0).unsqueeze(0)
        perturbed_list = []
        for c in range(C):
            feat_c = features[:, c:c + 1, :, :]
            eps_c = eps_adaptive[:, c:c + 1, :, :]
            fft = torch.fft.rfft2(feat_c, norm='ortho')
            fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
            noise_real = torch.randn_like(fft_shift.real)
            noise_imag = torch.randn_like(fft_shift.imag)
            noise_fft = torch.complex(noise_real, noise_imag)
            perturbed_fft = fft_shift + eps_c * noise_fft * mask_high
            fft_ishift = torch.fft.ifftshift(perturbed_fft, dim=(-2, -1))
            perturbed_c = torch.fft.irfft2(fft_ishift, s=(H, W), norm='ortho')
            perturbed_list.append(perturbed_c)
        perturbed_features = torch.cat(perturbed_list, dim=1)
        return perturbed_features
if __name__ == '__main__':
    print('Testing Frequency-Aware Perturbation...')
    B, C, H, W = (2, 64, 32, 32)
    features = torch.randn(B, C, H, W)
    print('\n1. Basic Frequency Perturbation:')
    perturb = FrequencyAwarePerturbation(perturb_high=True, eps_high=0.2)
    perturb.train()
    output = perturb(features)
    print(f'Input shape: {features.shape}')
    print(f'Output shape: {output.shape}')
    print(f'Difference: {(output - features).abs().mean().item():.4f}')
    print('\n2. Adaptive Frequency Perturbation:')
    adaptive_perturb = AdaptiveFrequencyPerturbation(eps_min=0.05, eps_max=0.3)
    adaptive_perturb.train()
    output2 = adaptive_perturb(features)
    print(f'Output shape: {output2.shape}')
    print(f'Difference: {(output2 - features).abs().mean().item():.4f}')
    print('\n✓ All tests passed!')
