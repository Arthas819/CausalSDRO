"""
    Data generator for inventory substitution.
"""

import torch
from torch.utils.data import *


class DataGenerator_Inventory:
    def __init__(self, n_train, n_test, inv_params, device):
        self.n_train, self.n_test = n_train, n_test
        self.d_x, self.d_y, self.d_z = inv_params['dimension']
        self.device = device
        self.n_m = inv_params.get('n_m', 1)
        self.seed = inv_params.get('seed', 42)
        self.generator = torch.Generator(device=device).manual_seed(self.seed)
        self.BETA_COEFF = torch.rand(self.d_x, device=device, dtype=torch.float64,
                                     generator=self.generator) * 0.2 - 0.1
        indices = torch.arange(self.d_x, device=device)
        covariance = 0.5 ** (indices[:, None] - indices[None, :]).abs().double()
        self.cholesky = torch.linalg.cholesky(covariance)
        print(f'DataGenerator: N={n_train}, D_X={self.d_x}, n_m={self.n_m}')

    def get_data(self, BATCH_SIZE):
        X_train_raw, Y_train = self.generate(self.n_train, self.n_m)
        X_test_raw, Y_test = self.generate(self.n_test)
        self.X_train_mean = X_train_raw.mean(dim=0, keepdim=True)
        self.X_train_std = X_train_raw.std(dim=0, keepdim=True) + 1e-8
        X_train = (X_train_raw - self.X_train_mean) / self.X_train_std
        X_test = (X_test_raw - self.X_train_mean) / self.X_train_std
        train_loader = DataLoader(TensorDataset(X_train, Y_train),
                                  batch_size=BATCH_SIZE, shuffle=True)
        return X_train, Y_train, X_test, Y_test, train_loader

    def generate(self, N, n_m=1):
        # M distinct features, n_m independent demands per feature.
        X = torch.randn(N // n_m, self.d_x, device=self.device, dtype=torch.float64,
                        generator=self.generator) @ self.cholesky.T
        X = X.repeat_interleave(n_m, dim=0)
        return X, self.generate_y(X)

    def generate_y(self, X, generator=None):
        return self.f_true(X @ self.BETA_COEFF, generator)

    def f_true(self, lam, generator=None):
        if generator is None:
            generator = self.generator
        scale = torch.exp(lam)
        uniforms = torch.rand((*lam.shape, 7), device=self.device, dtype=torch.float64,
                              generator=generator).clamp_min(torch.finfo(torch.float64).tiny)
        exponentials = -torch.log(uniforms)
        # Exp(rate=scale), Gamma(2, scale), Gamma(4, scale).
        return torch.stack((exponentials[..., 0] / scale,
                            exponentials[..., 1:3].sum(dim=-1) * scale,
                            exponentials[..., 3:].sum(dim=-1) * scale), dim=-1)
