"""
    This file includes the data generator for inventory substitution
"""

import torch
import torch.distributions as dist
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Data Generator -- Inventory
class DataGenerator_Inventory:

    def __init__(self, n_train, n_test, inv_params, device):

        # Input parameters
        self.n_train = n_train
        self.n_test = n_test
        self.d_x = inv_params['dimension'][0]
        self.d_y = inv_params['dimension'][1]
        self.d_z = inv_params['dimension'][2]
        self.device = device

        # Fixed parameters
        # beta ~ U (-0.1, 0.1)
        self.BETA_COEFF = torch.rand(self.d_x, device=device, dtype=torch.float64) * 0.2 - 0.1
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
            y_sample = self.f_true(lambda_val)
            # Reject samples that Y \le 0
            if y_sample.min() >= 0:
                X_list.append(x_sample)
                Y_list.append(y_sample)
                generated_count += 1
            else:
                print('Error!')

        X_tensor = torch.stack(X_list).to(self.device)
        Y_tensor = torch.stack(Y_list).to(self.device)

        return X_tensor, Y_tensor

    # True conditional mean function
    # For toy example
    def f_true(self, lam):
        param_val = torch.exp(lam)

        # y2 ~ Gamma(shape=2, scale=param_val)
        # rate = 1.0 / scale
        y2_dist = dist.Gamma(
            concentration=torch.tensor(2.0, device=self.device, dtype=torch.float64),
            rate=1.0 / param_val
        )
        y2 = y2_dist.sample()

        # y3 ~ Gamma(shape=4, scale=param_val)
        y3_dist = dist.Gamma(
            concentration=torch.tensor(4.0, device=self.device, dtype=torch.float64),
            rate=1.0 / param_val
        )
        y3 = y3_dist.sample()

        # y1 ~ Exp(rate = param_val)
        y1_dist = dist.Exponential(param_val)
        y1 = y1_dist.sample()

        return torch.stack([y1, y2, y3])