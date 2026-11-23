"""
    This file includes some calculation functions.
"""
import math
from time import perf_counter
import torch.nn as nn
from scipy.optimize import *
import numpy as np
import torch

from Optimizers.COPT_Optimizer import *

'''
    Get loss value. 
'''
def get_loss(z, y_train, pro_params):
    loss = 0
    if pro_params['problem_name'] == 'Newsvendor':
        loss = newsvendor_loss(z, y_train, pro_params['h_cost'], pro_params['b_cost'])

    elif pro_params['problem_name'] == 'Portfolio':
        d_z = pro_params['dimension'][2]
        z0_bound = pro_params.get('z0_bound', 5.0)
        B = pro_params.get('B', 100.0)
        z_decision, beta = parse_mv_output(z, d_z, z0_bound)
        loss = portfolio_loss_mv(z_decision, y_train, beta, pro_params['eta'],
                                 B, pro_params.get('a', B / 50), z0_bound)

    elif pro_params['problem_name'] == 'Inventory':
        loss = inventory_loss(z, y_train, pro_params)

    return loss


'''
    Loss calculator for testing set. 
'''
def get_test_loss(model, X_test, Y_test, pro_params):

    # pro_params is a dict sorting the problem paramters
    d_z = pro_params['dimension'][2]

    # Shift to evaluation mode
    model.eval()

    if pro_params['problem_name'] == 'Newsvendor':
        # Get ERM loss
        with torch.no_grad():
            z = model(X_test).reshape(-1, d_z) # Get: [n_train, d_z]
            loss = newsvendor_loss(z, Y_test, pro_params['h_cost'], pro_params['b_cost'])
            # Return mean loss as a tensor
            return loss.mean().item()

    elif pro_params['problem_name'] == 'Portfolio':
        with torch.no_grad():
            z = model(X_test).reshape(-1, d_z)  # Get: [n_train, d_z]
            z_decision, beta = parse_mv_output(z, d_z)
            # loss = portfolio_loss_mv(z_decision, Y_test, beta, pro_params['eta'])
            # Calculate real loss
            z_decision_w = z_decision.detach().cpu().numpy()
            return_value  = (pro_params['future_return'] * z_decision_w).sum(axis=1)
            loss, _, _, _, _ = metrics(return_value, pro_params['eta'])
            # Return mean loss as a tensor
            return loss.mean().item()

    elif pro_params['problem_name'] == 'Inventory':
        with torch.no_grad():
            z = model(X_test).reshape(-1, d_z)  # Get: [n_train, d_z]
            loss = inventory_loss(z, Y_test, pro_params)
            # Return mean loss as a tensor
            return loss.mean().item()

    else:
        return 0

'''
    Newsvendor loss
'''

def newsvendor_loss(z, y, h_cost, b_cost, B=10.0, a=0.2):
    # a = B / 50; retain the original softplus scale a = 1 / 5
    # z: [batch_size, d_z], y: [batch_size, d_y]
    if z.dim() == 1:
        z = z.unsqueeze(-1)
    m = nn.Softplus(beta=1 / a)
    holding  = h_cost * m(z - y)
    stockout = b_cost * m(y - z)
    loss = (holding + stockout).sum(dim=-1, keepdim=True)
    # Normalized softplus truncation in Appendix F.1
    return B * (1 - m(B - loss) / m(loss.new_tensor(B)))

# Conditional SAA approximation of the oracle loss, using the same loss as evaluation (for F_star)
@torch.no_grad()
def calculate_F_star_saa_news(data_gen, X_test, pro_params, N_oracle_x=200, M_oracle_y=1000):
    generator = torch.Generator(device=data_gen.device).manual_seed(data_gen.seed + 100000)
    X_oracle = X_test[:N_oracle_x] * data_gen.X_train_std + data_gen.X_train_mean
    h = pro_params['h_cost'].detach().cpu()
    b = pro_params['b_cost'].detach().cpu()
    oracle_losses = []

    for x in X_oracle:
        Y_saa = data_gen.generate_y(x.repeat(M_oracle_y, 1), generator).cpu()

        def objective(z):
            z_tensor = Y_saa.new_tensor([[z]])
            return newsvendor_loss(z_tensor, Y_saa, h, b).mean().item()

        # Beyond this bound every sample loss is increasing in z.
        z_upper = max(1e-8, Y_saa.max().item() + 0.2 * math.log(b.item() / h.item()))
        z_grid = np.linspace(0, z_upper, 101)
        values = [objective(z) for z in z_grid]
        best_loss = min(values)
        # The smooth truncation need not preserve convexity; refine all grid minima.
        for j in range(1, len(z_grid) - 1):
            if values[j] <= values[j-1] and values[j] <= values[j+1]:
                result = minimize_scalar(objective, bounds=(z_grid[j-1], z_grid[j+1]),
                                         method='bounded')
                best_loss = min(best_loss, result.fun)
        oracle_losses.append(best_loss)

    return float(np.mean(oracle_losses))

'''
    Portfolio loss
    Input: decision z, future return rate y, expected return beta, and weight eta
'''

def portfolio_loss_mv(z, y, beta, eta, B=100.0, a=2.0, z0_bound=5.0):
    # calculate total return value
    port_return = (y * z).sum(dim=1, keepdim=True)
    # Appendix F.3: loss - lower_B, where lower_B = -eta * z0_bound - eta**2 / 4.
    loss = (port_return - beta - eta / 2) ** 2 + eta * (z0_bound - beta)
    # Normalized softplus truncation T_B in [0, B].
    m = nn.Softplus(beta=1 / a)
    loss = B * (1 - m(B - loss) / m(loss.new_tensor(B)))
    # return: [batch]
    return loss.squeeze(-1)


'''
   [Portfolio] Get decision z and variance beta from the output of models (d_z = d_y + 1)
'''

def parse_mv_output(output, d_z, z0_bound=5.0):
    # Get portfolio decision z (outout: [batch, d_z])
    z_logits = output[..., : d_z - 1]
    # Use Normalization to satisfy the constraints
    z = torch.softmax(z_logits, dim=-1)
    # Estimated expected return
    beta = z0_bound * torch.tanh(output[..., d_z - 1:] / z0_bound)

    return z, beta


'''
    [Portfolio] Disjoint kNN groups of n_m historical observations.
'''

def group_portfolio_data(X_train, Y_train, n_m):
    if n_m < 1 or len(X_train) < n_m or len(X_train) != len(Y_train):
        raise ValueError('Portfolio groups require aligned observations and 1 <= n_m <= N.')
    # Remove the oldest remainder so every group contains exactly n_m observations.
    remainder = len(X_train) % n_m
    X_train = X_train[remainder:]
    Y_train = Y_train[remainder:]
    X_grouped = X_train.clone()
    group_index = torch.empty(len(X_train), dtype=torch.long, device=X_train.device)
    remaining = torch.arange(len(X_train), device=X_train.device)

    for m in range(len(X_train) // n_m):
        # The earliest remaining date is the center; distances use normalized features only.
        center = remaining[0]
        distance = ((X_train[remaining[1:]] - X_train[center]) ** 2).sum(dim=1)
        nearest = torch.argsort(distance, stable=True)[:n_m - 1] + 1
        selected = torch.cat((remaining[:1], remaining[nearest]))
        X_grouped[selected] = X_train[center]
        group_index[selected] = m
        remaining = remaining[~torch.isin(remaining, selected)]

    return X_grouped, Y_train, group_index


'''
    [Portfolio] Calculate the return of a decision in serval days,
    including mean, standard derivation, shape, and loss 
'''

def metrics(ret, eta):
    mean, std = ret.mean(), ret.std(ddof=1)
    sharpe = np.sqrt(252) * mean / std if std else np.nan

    losses = -ret
    var_loss = np.percentile(losses, 95)
    cvar = losses[losses >= var_loss].mean()

    var_term  = ((ret - mean)**2).sum() / 60
    mean_term = -eta * mean
    loss = var_term + mean_term

    return loss, sharpe, cvar, std, mean


def inventory_loss(z, y, pro_params):
    z, y = torch.broadcast_tensors(z, y)
    loss = DifferentiableCOPTSolver.apply(z, y, pro_params)
    B = pro_params.get('B', 50.0)
    a = pro_params.get('a', B / 50)
    m = nn.Softplus(beta=1 / a)
    return B * (1 - m(B - loss) / m(loss.new_tensor(B)))


def inventory_kernel_y(y, uniforms, epsilon, norm):
    # Inverse CDF of the kernel restricted to nonnegative demand.
    if norm == 1:
        lower = 0.5 * torch.exp(-y / epsilon)
        probability = lower + (1 - lower) * uniforms
        probability = probability.clamp(torch.finfo(y.dtype).eps, 1 - torch.finfo(y.dtype).eps)
        centered = probability - 0.5
        noise = -epsilon * centered.sign() * torch.log1p(-2 * centered.abs())
    else:
        sigma = math.sqrt(epsilon / 2)
        lower = torch.special.ndtr(-y / sigma)
        probability = lower + (1 - lower) * uniforms
        probability = probability.clamp(torch.finfo(y.dtype).eps, 1 - torch.finfo(y.dtype).eps)
        noise = sigma * torch.special.ndtri(probability)
    return (y + noise).clamp_min(0)


def calculate_F_star_theoretical_is(data_gen, X_test, Y_test, pro_params):
    # Conditional SAA with the same regularized, bounded loss.
    started = perf_counter()
    print('Computing Inventory F* by conditional SAA...', flush=True)
    N_oracle_x = min(pro_params.get('N_oracle_x', 40), len(X_test))
    M_oracle_y = pro_params.get('M_oracle_y', 100)
    generator = torch.Generator(device=data_gen.device).manual_seed(data_gen.seed + 100000)
    X_oracle = X_test[:N_oracle_x] * data_gen.X_train_std + data_gen.X_train_mean
    costs = dict(pro_params)
    for key in ('h_cost', 'b_cost', 'c_cost', 's_cost'):
        costs[key] = costs[key].cpu()
    z_bound = pro_params.get('z_bound', 30.0)
    oracle_losses = []
    for x in X_oracle:
        Y_saa = data_gen.generate_y(x.repeat(M_oracle_y, 1), generator).cpu()

        def objective(values):
            z = torch.tensor(values, dtype=torch.float64, requires_grad=True)
            loss = inventory_loss(z.expand(len(Y_saa), -1), Y_saa, costs).mean()
            gradient, = torch.autograd.grad(loss, z)
            return loss.item(), gradient.detach().numpy()

        # Truncation can destroy convexity; use several starting points.
        candidates = []
        for scale in (0.5, 1.0, 1.5):
            initial = np.clip(scale * Y_saa.mean(dim=0).numpy(), 0, z_bound)
            result = minimize(objective, initial, jac=True, method='L-BFGS-B',
                              bounds=[(0, z_bound)] * Y_saa.shape[1],
                              options={'maxiter': 60, 'ftol': 1e-9, 'gtol': 1e-6})
            candidates.extend([objective(initial)[0], result.fun])
        oracle_losses.append(min(candidates))
    value = float(np.mean(oracle_losses))
    print(f'Inventory F*: {value:.6f}, {perf_counter() - started:.2f}s', flush=True)
    return value
