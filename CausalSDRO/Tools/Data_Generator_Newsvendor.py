"""
    This file includes the data generator for newsvendor, according to Yang et al. (2022)
"""

import torch
from torch.utils.data import *
import numpy as np
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# Data Generator for Newsvendor
class DataGenerator_News:

    def __init__(self, n_train, n_test, news_params, device):

        # Input parameters
        self.n_train = n_train
        self.n_test = n_test
        self.d_x = news_params['dimension'][0]
        self.sigma = news_params['sigma']
        self.device = device
        self.n_m = news_params.get('n_m', 1)
        self.seed = news_params.get('seed', 42)
        self.generator = torch.Generator(device=device).manual_seed(self.seed)

        if self.n_m < 1 or n_train % self.n_m != 0:
            raise ValueError('n_train must be divisible by n_m.')

        # Fixed parameters
        self.c = 1.7
        # beta ~ U (-0.1, 0.1)
        self.BETA_COEFF = torch.rand(self.d_x, 1, device=device, dtype=torch.float64,
                                     generator=self.generator) * 0.2 - 0.1
        # Covariance matrix for vector x
        self.cov = 0.5
        print(f"DataGenerator: N_train={self.n_train}, N_test={self.n_test}, D_X={self.d_x}, n_m={self.n_m}")

    # Get all training and testing data and a loader for training
    def get_data(self, BATCH_SIZE):
        # Get data
        X_train_row, Y_train = self.generate(self.n_train, self.n_m)
        X_test_row, Y_test = self.generate(self.n_test)

        # Normalization for training features
        X_train_mean = X_train_row.mean(dim=0, keepdim=True)
        X_train_std = X_train_row.std(dim=0, keepdim=True) + 1e-8
        self.X_train_mean = X_train_mean
        self.X_train_std = X_train_std
        X_train = (X_train_row - X_train_mean) / X_train_std

        # Normalization for testing features
        # ----Notice! X_test need to be normalized by mean and std of training dataset!
        X_test = (X_test_row - X_train_mean) / X_train_std

        # Set data loader
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        return X_train, Y_train, X_test, Y_test, train_loader

    # Genrate our data!
    def generate(self, N, n_m=1):
        # Generate index list
        indices = np.arange(self.d_x)
        # covariance matrix
        cov_matrix = self.cov ** np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])
        # To tensor
        cov_matrix_tensor = torch.tensor(cov_matrix, dtype=torch.float64).to(self.device)
        # M = N / n_m distinct covariates; n_m demands for each covariate
        X_tensor = torch.randn(N // n_m, self.d_x, device=self.device, dtype=torch.float64,
                               generator=self.generator) @ torch.linalg.cholesky(cov_matrix_tensor).T
        X_tensor = X_tensor.repeat_interleave(n_m, dim=0)
        Y_tensor = self.generate_y(X_tensor)

        return X_tensor, Y_tensor

    # Sample positive conditional demands, keeping the covariate fixed
    def generate_y(self, X, generator=None):
        if generator is None:
            generator = self.generator
        y_origin = self.f_true(X @ self.BETA_COEFF)
        Y = y_origin + self.sigma * torch.randn(y_origin.shape, device=self.device,
                                                dtype=torch.float64, generator=generator)
        rejected = Y <= 0
        while rejected.any():
            Y[rejected] = y_origin[rejected] + self.sigma * torch.randn(
                int(rejected.sum().item()), device=self.device, dtype=torch.float64, generator=generator)
            rejected = Y <= 0
        return Y

    # True conditional mean function
    def f_true(self, lam):
        return self.c * (torch.sin(2 * lam) + 2 * torch.exp(-16 * lam ** 2) + 1)