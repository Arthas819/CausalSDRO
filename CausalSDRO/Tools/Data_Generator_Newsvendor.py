"""
    This file includes the data generator for newsvendor, according to Yang et al. (2022)
"""

import torch
import torch.distributions as dist
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Data Generator for Newsvendor in Sec 5.1
class DataGenerator_News:

    def __init__(self, n_train, n_test, news_params, device):

        # Input parameters
        self.n_train = n_train
        self.n_test = n_test
        self.d_x = news_params['dimension'][0]
        self.sigma = news_params['sigma']
        self.device = device

        # Fixed parameters
        self.c = 1.7
        # beta ~ U (-0.1, 0.1)
        self.BETA_COEFF = torch.rand(self.d_x, 1, device=device, dtype=torch.float64) * 0.2 - 0.1
        # Covariance matrix for vector x
        self.cov = 0.5
        print(f"DataGenerator: N_train={self.n_train}, N_test={self.n_test}, D_X={self.d_x}")

    # Get all training and testing data and a loader for training
    def get_data(self, BATCH_SIZE):
        # Get data
        X_train_row, Y_train = self.generate(self.n_train)
        X_test_row, Y_test = self.generate(self.n_test)

        # Normalization for training features
        X_train_mean = X_train_row.mean(dim=0, keepdim=True)
        X_train_std = X_train_row.std(dim=0, keepdim=True) + 1e-8
        X_train = (X_train_row - X_train_mean) / X_train_std

        # Normalization for testing features
        # ----Notice! X_test need to be normalized by mean and std of training dataset!
        X_test = (X_test_row - X_train_mean) / X_train_std

        # Set data loader
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        return X_train, Y_train, X_test, Y_test, train_loader

    # Genrate our data!
    def generate(self, N):
        # Generate index list
        indices = np.arange(self.d_x)
        # covariance matrix
        cov_matrix = self.cov ** np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])
        # To tensor
        cov_matrix_tensor = torch.tensor(cov_matrix, dtype=torch.float64).to(self.device)
        # Create a distribution by PyTorch-distribution, X ~ N(0, Sigma)
        mvn = dist.MultivariateNormal(torch.zeros(self.d_x, device=self.device, dtype=torch.float64),
                                      covariance_matrix=cov_matrix_tensor)
        # Sample from this list
        X_list, Y_list = [], []
        generated_count = 0
        while generated_count < N:
            # Get X
            x_sample = mvn.sample()
            lambda_val = x_sample @ self.BETA_COEFF
            # Get f_true(beta x)
            y_origin = self.f_true(lambda_val)
            # Create noise N(0, sigma^2)
            noise = torch.normal(0.0, self.sigma, (1,), device=self.device, dtype=torch.float64)
            # Create Y
            y_sample = y_origin + noise
            # Reject samples that Y \le 0
            if y_sample > 0:
                X_list.append(x_sample)
                Y_list.append(y_sample)
                generated_count += 1

        X_tensor = torch.stack(X_list).to(self.device)
        Y_tensor = torch.stack(Y_list).to(self.device)

        return X_tensor, Y_tensor

    # True conditional mean function
    def f_true(self, lam):
        return self.c * (torch.sin(2 * lam) + 2 * torch.exp(-16 * lam ** 2) + 1)