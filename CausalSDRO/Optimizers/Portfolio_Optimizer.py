"""
    This file solves portfolio MV models.
"""

import copy
import math
from pathlib import *
from Tools.Functions import *
from Optimizers.Smooth_Trainer import *
from Optimizers.KLDRO_Trainer import *
from Optimizers.CWDRO_Trainer import *
import cvxpy as cp
import pandas as pd
from torch.utils.data import *

from Tools.Decision_Rules import *

from Optimizers.ERM_Trainer import *
from Optimizers.CSDRO_Trainer import *


'''
    Solve normal mean-variance model
'''

def normal_mv_solver(historical_returns, ETA):
    # Input DataFrame, turn to Numpy
    T, n = historical_returns.shape
    # Unconditional Estimation
    mu_hat = np.mean(historical_returns, axis=0)
    Sigma_hat = np.cov(historical_returns, rowvar=False) + 1e-6 * np.eye(n)

    # Optimizer
    w_var = cp.Variable(n)
    # Objectitve
    risk_term = cp.quad_form(w_var, Sigma_hat)
    return_term = ETA * (mu_hat @ w_var)
    objective = cp.Minimize(risk_term - return_term)
    # constraints
    constraints = [
        cp.sum(w_var) == 1,
        w_var >= 0
    ]
    prob = cp.Problem(objective, constraints)
    # solve
    try:
        prob.solve(solver=cp.GUROBI)
        # prob.solve(solver=cp.OSQP)
    except Exception:
        prob.solve()
    # Results
    if prob.status != 'optimal':
        print(f"MV Solver Warning: Status is {prob.status}, returning Equal Weights.")
        return np.ones(n) / n
    w_opt = w_var.value
    w_opt[w_opt < 0] = 0
    w_opt = w_opt / np.sum(w_opt)

    return w_opt


'''
    Solve post_hc mean-variance model
'''

def post_hc_solver(future_return, ETA):
    future_return = np.asarray(future_return)
    T, n = future_return.shape
    mu = future_return.mean(axis=0)
    centered = future_return - mu
    w_var = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(centered @ w_var) / 60 - ETA * (mu @ w_var))
    prob = cp.Problem(objective, [cp.sum(w_var) == 1, w_var >= 0])
    try:
        prob.solve(solver=cp.GUROBI)
    except Exception:
        prob.solve(solver=cp.CLARABEL)
    if prob.status not in ('optimal', 'optimal_inaccurate'):
        raise RuntimeError(f"Post-hoc solver failed: {prob.status}")
    return w_var.value


def solve_all_models(date, x_features, y_ret_select, q_params, f_params, device, tnn_iterations=1200):
    # Input: date, features, returns, parameters (for question and fixed)
    best_parameters = pd.read_excel(Path(__file__).resolve().parents[1] /
                                    'Outputs/Outputs_Tuning/DRO_Portfolio_Best_Parameters.xlsx')
    if 'n_m' in best_parameters.columns:
        best_parameters = best_parameters[(best_parameters['ETA'] == q_params['eta']) &
                                          (best_parameters['n_m'] == q_params['n_m'])]
    else:
        best_parameters = best_parameters[(best_parameters['ETA'] == q_params['eta']) |
                                          (best_parameters['Model'] == 'ERM')]

    # Results
    records = []

    # History features
    x_historical_1 = x_features.loc[x_features.index < date, :]
    x_historical_row = x_historical_1.iloc[-q_params['historical_days']:, :].tail(q_params.get('train_days', 730))
    # Normalization
    scaler_mean = x_historical_row.mean()
    scaler_std = x_historical_row.std()
    x_historical = (x_historical_row - scaler_mean) / (scaler_std + 1e-8)

    # Feature Today
    x_today_row = x_features.loc[date:, :].iloc[:1, :]
    # Normalization (using mean and std of training set)
    x_today = (x_today_row - scaler_mean) / (scaler_std + 1e-8)

    # History return
    y_historical_1 = y_ret_select.loc[y_ret_select.index < date, :]
    y_historical = y_historical_1.iloc[-q_params['historical_days']:, :].tail(q_params.get('train_days', 730))

    # Future return
    future_return = y_ret_select[date:].head(60)

    # Transform to Tensor
    x_historical_tensor = torch.tensor(x_historical.values, dtype=torch.float64).to(device)
    y_historical_tensor = torch.tensor(y_historical.values, dtype=torch.float64).to(device)
    x_today_tensor = torch.tensor(x_today.values, dtype=torch.float64).to(device)

    # Disjoint kNN groups of normalized historical features.
    n_m = q_params['n_m']
    x_historical_tensor, y_historical_tensor, group_index = group_portfolio_data(
        x_historical_tensor, y_historical_tensor, n_m)
    q_params['group_index'] = group_index
    print(f"Groups: n_m={n_m}, M={len(x_historical_tensor) // n_m}, N={len(x_historical_tensor)}")

    # parameters
    eta = q_params['eta']

    ## ===== Baselines =====
    # Equal Weight Method
    w_eq = np.full(y_historical.shape[1], 1 / y_historical.shape[1])
    ret_eq = (future_return * w_eq).sum(axis=1)
    # Standard mean-variance model
    w_nmv = normal_mv_solver(y_historical.values, eta)
    return_nmv = (future_return * w_nmv).sum(axis=1)
    # Post-hoc (Complete Information) Method
    weight_post_hc = post_hc_solver(future_return.values, eta)
    return_post_hc = (future_return * weight_post_hc).sum(axis=1)

    ## ===== Add features, CMV =====
    # Preparation
    d_x = x_features.shape[1]
    d_y = y_historical.shape[1]
    d_z = d_y + 1

    # Upload data batch
    train_dataset = TensorDataset(x_historical_tensor, y_historical_tensor)
    train_loader = DataLoader(train_dataset, batch_size=f_params.BATCH_SIZE, shuffle=False)

    K = f_params.EPOCHS_SCSC * len(train_loader)
    f_params.setLearningRate(1 / np.sqrt(K), 2 / np.sqrt(K))
    erm_row = best_parameters[best_parameters['Model'] == 'ERM'].iloc[0]
    f_params.setCVParameters(erm_row, len(train_loader))

    # Solve the Conditional Stochastic MV by SAA and SCSC
    # Train CMV with TNN by SAA
    print('Training Conditional Stochastic MV by ERM')
    # Model -> float 64
    np.random.seed(42)
    torch.manual_seed(42)
    model_cmv_tnn_saa_obj = TNN(input_dim=d_x, hidden_dim=f_params.hidden_layer_dim, output_dim=d_z).to(device).double()
    # Train this model as ERM, return a trained model and a historical loss list for drawing convergence curve
    erm_trainer = ERM_Trainer(model_cmv_tnn_saa_obj, train_loader, f_params, device, 'TNN')
    model_cmv_tnn_saa, _ = erm_trainer.train_erm(x_historical_tensor, y_historical_tensor, q_params)

    # Get weight and feature return
    w_cmv_tnn_saa, beta = parse_mv_output(model_cmv_tnn_saa(x_today_tensor), d_z)
    w_cmv_tnn_saa = w_cmv_tnn_saa.detach().cpu().numpy()
    return_cmv_tnn_saa = (future_return * w_cmv_tnn_saa).sum(axis=1)

    # Train CMV with SRF by SAA
    np.random.seed(42)
    torch.manual_seed(42)
    model_cmv_srf_saa_obj = SRF(input_dim=d_x, depth=f_params.SRF_depth, output_dim=d_z,
                                    tree_number=f_params.Tree_number, params = f_params).to(device).double()

    erm_trainer = ERM_Trainer(model_cmv_srf_saa_obj, train_loader, f_params, device, 'SRF')
    model_cmv_srf_saa, _ = erm_trainer.train_erm(x_historical_tensor, y_historical_tensor, q_params)

    # Get weight and feature return
    w_cmv_srf_saa, beta = parse_mv_output(model_cmv_srf_saa(x_today_tensor), d_z)
    w_cmv_srf_saa = w_cmv_srf_saa.detach().cpu().numpy()
    return_cmv_srf_saa = (future_return * w_cmv_srf_saa).sum(axis=1)

    # CMV is ERM (lambda=epsilon=0); retain the old output column names.
    return_cmv_tnn_scsc = return_cmv_tnn_saa
    return_cmv_srf_scsc = return_cmv_srf_saa

    ## ===== DRO models ======
    # For 1-CausalSDRO-MV (p=1), TNN &SRF, SAA & SCSC

    # Best parameter combination
    best_row = best_parameters[
        (best_parameters['Model'] == 'Causal-SDRO') & (best_parameters['norm'] == 1)]
    q_params['lambda'] = best_row.iloc[0]['best_lambda']
    q_params['epsilon'] = best_row.iloc[0]['best_epsilon']
    q_params['norm'] = 1
    f_params.setCVParameters(best_row.iloc[0], len(train_loader))
    # Train 1-CausalSDRO-MV with TNN by SAA
    # Model
    np.random.seed(42)
    torch.manual_seed(42)
    model_1_sdmv_tnn_saa_obj = TNN(input_dim=d_x, hidden_dim=f_params.hidden_layer_dim, output_dim=d_z).to(device)
    # float 64
    model_1_sdmv_tnn_saa_obj = model_1_sdmv_tnn_saa_obj.double()
    # train
    erm_trainer = ERM_Trainer(model_1_sdmv_tnn_saa_obj, train_loader, f_params, device, 'TNN')
    model_1_sdmv_tnn_saa, _ = erm_trainer.train_erm(x_historical_tensor, y_historical_tensor, q_params)

    # Get weight and feature return
    w_1_sdmv_tnn_saa, beta = parse_mv_output(model_1_sdmv_tnn_saa(x_today_tensor), d_z)
    w_1_sdmv_tnn_saa = w_1_sdmv_tnn_saa.detach().cpu().numpy()
    return_1_sdmv_tnn_saa = (future_return * w_1_sdmv_tnn_saa).sum(axis=1)


    # Train 1-CausalSDRO-MV with SRF by SAA
    np.random.seed(42)
    torch.manual_seed(42)
    model_1_sdmv_srf_saa_obj = SRF(input_dim=d_x, depth=f_params.SRF_depth, output_dim=d_z,
                                    tree_number=f_params.Tree_number, params = f_params).to(device)
    # float 64
    model_1_sdmv_srf_saa_obj = model_1_sdmv_srf_saa_obj.double()
    # train
    erm_trainer = ERM_Trainer(model_1_sdmv_srf_saa_obj, train_loader, f_params, device, 'SRF')
    model_1_sdmv_srf_saa, _ = erm_trainer.train_erm(x_historical_tensor, y_historical_tensor, q_params)

    # Get weight and feature return
    w_1_sdmv_srf_saa, beta = parse_mv_output(model_1_sdmv_srf_saa(x_today_tensor), d_z)
    w_1_sdmv_srf_saa = w_1_sdmv_srf_saa.detach().cpu().numpy()
    return_1_sdmv_srf_saa = (future_return * w_1_sdmv_srf_saa).sum(axis=1)

    # Now, SCSC
    print('Training 1-CausalSDRO MV by SCSC')
    # Train 1-CausalSDRO-MV with TNN by SCSC
    np.random.seed(42)
    torch.manual_seed(42)
    model_1_sdmv_tnn_scsc_obj = TNN(input_dim=d_x, hidden_dim=f_params.hidden_layer_dim, output_dim=d_z).to(device)
    model_1_sdmv_tnn_scsc_obj = model_1_sdmv_tnn_scsc_obj.double()
    # Train the DRO model by SCSC
    # Override only the TNN update budget; retain the selected step-size scales and Q.
    tnn_params = copy.copy(f_params)
    tnn_row = best_row.iloc[0].copy()
    tnn_row['K'] = tnn_iterations
    tnn_params.setCVParameters(tnn_row, len(train_loader))
    scsc_trainer = CSDRO_Trainer(model_1_sdmv_tnn_scsc_obj, train_loader, tnn_params, device, 'TNN')
    # Return a trained model and a historical loss list
    model_1_sdmv_tnn_scsc, _ = scsc_trainer.train_scsc(x_historical_tensor, y_historical_tensor, q_params)

    # Get weight and feature return
    w_1_sdmv_tnn_scsc, beta = parse_mv_output(model_1_sdmv_tnn_scsc(x_today_tensor), d_z)
    w_1_sdmv_tnn_scsc = w_1_sdmv_tnn_scsc.detach().cpu().numpy()
    return_1_sdmv_tnn_scsc = (future_return * w_1_sdmv_tnn_scsc).sum(axis=1)


    # Train 1-CausalSDRO-MV with SRF by SCSC
    np.random.seed(42)
    torch.manual_seed(42)
    model_1_sdmv_srf_scsc_obj = SRF(input_dim=d_x, depth=f_params.SRF_depth, output_dim=d_z,
                                 tree_number = f_params.Tree_number, params = f_params).to(device)
    model_1_sdmv_srf_scsc_obj = model_1_sdmv_srf_scsc_obj.double()

    # Train the DRO model by SCSC
    scsc_trainer = CSDRO_Trainer(model_1_sdmv_srf_scsc_obj, train_loader, f_params, device, 'SRF')
    # Return a trained model and a historical loss list
    model_1_sdmv_srf_scsc, _ = scsc_trainer.train_scsc(x_historical_tensor, y_historical_tensor, q_params)

    # Get weight and feature return
    w_1_sdmv_srf_scsc, beta = parse_mv_output(model_1_sdmv_srf_scsc(x_today_tensor), d_z)
    w_1_sdmv_srf_scsc = w_1_sdmv_srf_scsc.detach().cpu().numpy()
    return_1_sdmv_srf_scsc = (future_return * w_1_sdmv_srf_scsc).sum(axis=1)

    ## For 2-CausalSDRO-MV (p=2), TNN &SRF, SAA & SCSC
    # Best parameter combination
    best_row = best_parameters[
        (best_parameters['Model'] == 'Causal-SDRO') & (best_parameters['norm'] == 2)]
    q_params['lambda'] = best_row.iloc[0]['best_lambda']
    q_params['epsilon'] = best_row.iloc[0]['best_epsilon']
    q_params['norm'] = 2
    f_params.setCVParameters(best_row.iloc[0], len(train_loader))
    # Train 2-CausalSDRO-MV with TNN by SAA
    # Model
    np.random.seed(42)
    torch.manual_seed(42)
    model_2_sdmv_tnn_saa_obj = TNN(input_dim=d_x, hidden_dim=f_params.hidden_layer_dim, output_dim=d_z).to(device)
    # float 64
    model_2_sdmv_tnn_saa_obj = model_2_sdmv_tnn_saa_obj.double()
    # train
    erm_trainer = ERM_Trainer(model_2_sdmv_tnn_saa_obj, train_loader, f_params, device, 'TNN')
    model_2_sdmv_tnn_saa, _ = erm_trainer.train_erm(x_historical_tensor, y_historical_tensor, q_params)

    # Get weight and feature return
    w_2_sdmv_tnn_saa, beta = parse_mv_output(model_2_sdmv_tnn_saa(x_today_tensor), d_z)
    w_2_sdmv_tnn_saa = w_2_sdmv_tnn_saa.detach().cpu().numpy()
    return_2_sdmv_tnn_saa = (future_return * w_2_sdmv_tnn_saa).sum(axis=1)

    # Train 2-CausalSDRO-MV with SRF by SAA
    np.random.seed(42)
    torch.manual_seed(42)
    model_2_sdmv_srf_saa_obj = SRF(input_dim=d_x, depth=f_params.SRF_depth, output_dim=d_z,
                                    tree_number=f_params.Tree_number, params = f_params).to(device)
    # float 64
    model_2_sdmv_srf_saa_obj = model_2_sdmv_srf_saa_obj.double()

    # train
    erm_trainer = ERM_Trainer(model_2_sdmv_srf_saa_obj, train_loader, f_params, device, 'SRF')
    model_2_sdmv_srf_saa, _ = erm_trainer.train_erm(x_historical_tensor, y_historical_tensor, q_params)

    # Get weight and feature return
    w_2_sdmv_srf_saa, beta = parse_mv_output(model_2_sdmv_srf_saa(x_today_tensor), d_z)
    w_2_sdmv_srf_saa = w_2_sdmv_srf_saa.detach().cpu().numpy()
    return_2_sdmv_srf_saa = (future_return * w_2_sdmv_srf_saa).sum(axis=1)

    # print(w_2_sdmv_srf_saa)

    # Now, SCSC
    print('Training 2-CausalSDRO MV by SCSC')
    # Train 2-CausalSDRO-MV with TNN by SCSC
    np.random.seed(42)
    torch.manual_seed(42)
    model_2_sdmv_tnn_scsc_obj = TNN(input_dim=d_x, hidden_dim=f_params.hidden_layer_dim, output_dim=d_z).to(device)
    model_2_sdmv_tnn_scsc_obj = model_2_sdmv_tnn_scsc_obj.double()
    # Train the DRO model by SCSC
    # Override only the TNN update budget; retain the selected step-size scales and Q.
    tnn_params = copy.copy(f_params)
    tnn_row = best_row.iloc[0].copy()
    tnn_row['K'] = tnn_iterations
    tnn_params.setCVParameters(tnn_row, len(train_loader))
    scsc_trainer = CSDRO_Trainer(model_2_sdmv_tnn_scsc_obj, train_loader, tnn_params, device, 'TNN')
    # Return a trained model and a historical loss list
    model_2_sdmv_tnn_scsc, _ = scsc_trainer.train_scsc(x_historical_tensor, y_historical_tensor, q_params)

    # Get weight and feature return
    w_2_sdmv_tnn_scsc, beta = parse_mv_output(model_2_sdmv_tnn_scsc(x_today_tensor), d_z)
    w_2_sdmv_tnn_scsc = w_2_sdmv_tnn_scsc.detach().cpu().numpy()
    return_2_sdmv_tnn_scsc = (future_return * w_2_sdmv_tnn_scsc).sum(axis=1)

    # Train 2-CausalSDRO-MV with SRF by SCSC
    np.random.seed(42)
    torch.manual_seed(42)
    model_2_sdmv_srf_scsc_obj = SRF(input_dim=d_x, depth=f_params.SRF_depth, output_dim=d_z,
                                 tree_number=f_params.Tree_number, params = f_params).to(device)
    model_2_sdmv_srf_scsc_obj = model_2_sdmv_srf_scsc_obj.double()

    # Train the DRO model by SCSC
    scsc_trainer = CSDRO_Trainer(model_2_sdmv_srf_scsc_obj, train_loader, f_params, device, 'SRF')
    # Return a trained model and a historical loss list
    model_2_sdmv_srf_scsc, _ = scsc_trainer.train_scsc(x_historical_tensor, y_historical_tensor, q_params)

    # Get weight and feature return
    w_2_sdmv_srf_scsc, beta = parse_mv_output(model_2_sdmv_srf_scsc(x_today_tensor), d_z)
    w_2_sdmv_srf_scsc = w_2_sdmv_srf_scsc.detach().cpu().numpy()
    return_2_sdmv_srf_scsc = (future_return * w_2_sdmv_srf_scsc).sum(axis=1)

    # print(w_2_sdmv_srf_scsc)

    ## ===== Baselines =====

    # Summary all results!
    # baselines
    loss_ph, sharpe_ph, cvar_ph, var_ph, mean_ph = metrics(return_post_hc, eta)
    loss_eq, sharpe_eq, cvar_eq, var_eq, mean_eq = metrics(ret_eq, eta)
    loss_nmv, sharpe_nmv, cvar_nmv, var_nmv, mean_nmv = metrics(return_nmv, eta)
    # With features
    loss_cmv_tnn_saa, sharpe_cmv_tnn_saa, cvar_cmv_tnn_saa, var_cmv_tnn_saa, mean_cmv_tnn_saa = metrics(return_cmv_tnn_saa, eta)
    loss_cmv_tnn_scsc, sharpe_cmv_tnn_scsc, cvar_cmv_tnn_scsc, var_cmv_tnn_scsc, mean_cmv_tnn_scsc = metrics(return_cmv_tnn_scsc, eta)
    loss_cmv_srf_saa, sharpe_cmv_srf_saa, cvar_cmv_srf_saa, var_cmv_srf_saa, mean_cmv_srf_saa = metrics(return_cmv_srf_saa, eta)
    loss_cmv_srf_scsc, sharpe_cmv_srf_scsc, cvar_cmv_srf_scsc, var_cmv_srf_scsc, mean_cmv_srf_scsc = metrics(return_cmv_srf_scsc, eta)
    # DRO models
    # p = 1
    loss_1_sdmv_tnn_saa, sharpe_1_sdmv_tnn_saa, cvar_1_sdmv_tnn_saa, var_1_sdmv_tnn_saa, mean_1_sdmv_tnn_saa = metrics(return_1_sdmv_tnn_saa, eta)
    loss_1_sdmv_tnn_scsc, sharpe_1_sdmv_tnn_scsc, cvar_1_sdmv_tnn_scsc, var_1_sdmv_tnn_scsc, mean_1_sdmv_tnn_scsc = metrics(return_1_sdmv_tnn_scsc, eta)
    loss_1_sdmv_srf_saa, sharpe_1_sdmv_srf_saa, cvar_1_sdmv_srf_saa, var_1_sdmv_srf_saa, mean_1_sdmv_srf_saa = metrics(return_1_sdmv_srf_saa, eta)
    loss_1_sdmv_srf_scsc, sharpe_1_sdmv_srf_scsc, cvar_1_sdmv_srf_scsc, var_1_sdmv_srf_scsc, mean_1_sdmv_srf_scsc = metrics(return_1_sdmv_srf_scsc, eta)
    # p = 2
    loss_2_sdmv_tnn_saa, sharpe_2_sdmv_tnn_saa, cvar_2_sdmv_tnn_saa, var_2_sdmv_tnn_saa, mean_2_sdmv_tnn_saa = metrics(return_2_sdmv_tnn_saa, eta)
    loss_2_sdmv_tnn_scsc, sharpe_2_sdmv_tnn_scsc, cvar_2_sdmv_tnn_scsc, var_2_sdmv_tnn_scsc, mean_2_sdmv_tnn_scsc = metrics(return_2_sdmv_tnn_scsc, eta)
    loss_2_sdmv_srf_saa, sharpe_2_sdmv_srf_saa, cvar_2_sdmv_srf_saa, var_2_sdmv_srf_saa, mean_2_sdmv_srf_saa = metrics(return_2_sdmv_srf_saa, eta)
    loss_2_sdmv_srf_scsc, sharpe_2_sdmv_srf_scsc, cvar_2_sdmv_srf_scsc, var_2_sdmv_srf_scsc, mean_2_sdmv_srf_scsc = metrics(return_2_sdmv_srf_scsc, eta)

    # Input all results
    records.append({
        # ----- Genral -----
        'date': date,
        'eta': eta,
        'n_m': n_m, 'N': len(x_historical_tensor), 'M': len(x_historical_tensor) // n_m,

        # ----- baselines -----
        'loss_ph': loss_ph,
        'sharpe_ph': sharpe_ph,
        'cvar_ph': cvar_ph,
        'var_ph': var_ph,
        'mean_ph': mean_ph,

        'loss_eq': loss_eq,
        'sharpe_eq': sharpe_eq,
        'cvar_eq': cvar_eq,
        'var_eq': var_eq,
        'mean_eq': mean_eq,

        'loss_nmv': loss_nmv,
        'sharpe_nmv': sharpe_nmv,
        'cvar_nmv': cvar_nmv,
        'var_nmv': var_nmv,
        'mean_nmv': mean_nmv,

        # ----- With features -----
        'loss_cmv_tnn_saa': loss_cmv_tnn_saa,
        'sharpe_cmv_tnn_saa': sharpe_cmv_tnn_saa,
        'cvar_cmv_tnn_saa': cvar_cmv_tnn_saa,
        'var_cmv_tnn_saa': var_cmv_tnn_saa,
        'mean_cmv_tnn_saa': mean_cmv_tnn_saa,

        'loss_cmv_tnn_scsc': loss_cmv_tnn_scsc,
        'sharpe_cmv_tnn_scsc': sharpe_cmv_tnn_scsc,
        'cvar_cmv_tnn_scsc': cvar_cmv_tnn_scsc,
        'var_cmv_tnn_scsc': var_cmv_tnn_scsc,
        'mean_cmv_tnn_scsc': mean_cmv_tnn_scsc,

        'loss_cmv_srf_saa': loss_cmv_srf_saa,
        'sharpe_cmv_srf_saa': sharpe_cmv_srf_saa,
        'cvar_cmv_srf_saa': cvar_cmv_srf_saa,
        'var_cmv_srf_saa': var_cmv_srf_saa,
        'mean_cmv_srf_saa': mean_cmv_srf_saa,

        'loss_cmv_srf_scsc': loss_cmv_srf_scsc,
        'sharpe_cmv_srf_scsc': sharpe_cmv_srf_scsc,
        'cvar_cmv_srf_scsc': cvar_cmv_srf_scsc,
        'var_cmv_srf_scsc': var_cmv_srf_scsc,
        'mean_cmv_srf_scsc': mean_cmv_srf_scsc,

        # ----- DRO p=1 -----
        'loss_1_sdmv_tnn_saa': loss_1_sdmv_tnn_saa,
        'sharpe_1_sdmv_tnn_saa': sharpe_1_sdmv_tnn_saa,
        'cvar_1_sdmv_tnn_saa': cvar_1_sdmv_tnn_saa,
        'var_1_sdmv_tnn_saa': var_1_sdmv_tnn_saa,
        'mean_1_sdmv_tnn_saa': mean_1_sdmv_tnn_saa,

        'loss_1_sdmv_tnn_scsc': loss_1_sdmv_tnn_scsc,
        'sharpe_1_sdmv_tnn_scsc': sharpe_1_sdmv_tnn_scsc,
        'cvar_1_sdmv_tnn_scsc': cvar_1_sdmv_tnn_scsc,
        'var_1_sdmv_tnn_scsc': var_1_sdmv_tnn_scsc,
        'mean_1_sdmv_tnn_scsc': mean_1_sdmv_tnn_scsc,

        'loss_1_sdmv_srf_saa': loss_1_sdmv_srf_saa,
        'sharpe_1_sdmv_srf_saa': sharpe_1_sdmv_srf_saa,
        'cvar_1_sdmv_srf_saa': cvar_1_sdmv_srf_saa,
        'var_1_sdmv_srf_saa': var_1_sdmv_srf_saa,
        'mean_1_sdmv_srf_saa': mean_1_sdmv_srf_saa,

        'loss_1_sdmv_srf_scsc': loss_1_sdmv_srf_scsc,
        'sharpe_1_sdmv_srf_scsc': sharpe_1_sdmv_srf_scsc,
        'cvar_1_sdmv_srf_scsc': cvar_1_sdmv_srf_scsc,
        'var_1_sdmv_srf_scsc': var_1_sdmv_srf_scsc,
        'mean_1_sdmv_srf_scsc': mean_1_sdmv_srf_scsc,

        # ----- DRO p=2 -----
        'loss_2_sdmv_tnn_saa': loss_2_sdmv_tnn_saa,
        'sharpe_2_sdmv_tnn_saa': sharpe_2_sdmv_tnn_saa,
        'cvar_2_sdmv_tnn_saa': cvar_2_sdmv_tnn_saa,
        'var_2_sdmv_tnn_saa': var_2_sdmv_tnn_saa,
        'mean_2_sdmv_tnn_saa': mean_2_sdmv_tnn_saa,

        'loss_2_sdmv_tnn_scsc': loss_2_sdmv_tnn_scsc,
        'sharpe_2_sdmv_tnn_scsc': sharpe_2_sdmv_tnn_scsc,
        'cvar_2_sdmv_tnn_scsc': cvar_2_sdmv_tnn_scsc,
        'var_2_sdmv_tnn_scsc': var_2_sdmv_tnn_scsc,
        'mean_2_sdmv_tnn_scsc': mean_2_sdmv_tnn_scsc,

        'loss_2_sdmv_srf_saa': loss_2_sdmv_srf_saa,
        'sharpe_2_sdmv_srf_saa': sharpe_2_sdmv_srf_saa,
        'cvar_2_sdmv_srf_saa': cvar_2_sdmv_srf_saa,
        'var_2_sdmv_srf_saa': var_2_sdmv_srf_saa,
        'mean_2_sdmv_srf_saa': mean_2_sdmv_srf_saa,

        'loss_2_sdmv_srf_scsc': loss_2_sdmv_srf_scsc,
        'sharpe_2_sdmv_srf_scsc': sharpe_2_sdmv_srf_scsc,
        'cvar_2_sdmv_srf_scsc': cvar_2_sdmv_srf_scsc,
        'var_2_sdmv_srf_scsc': var_2_sdmv_srf_scsc,
        'mean_2_sdmv_srf_scsc': mean_2_sdmv_srf_scsc,
    })

    return records

'''
    Compare all portfolio models for one date, omega, and grouping size.
'''

def portfolio_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PortfolioTrackingBatch(CSDRO_Trainer):
    # Average group gradients with a shared pair of independent kernel draws.
    def train_step_batch(self):
        if self.beta_k == 1:
            return super().train_step_batch()
        self.model.train()
        groups = torch.multinomial(self.p_m, int(self.params.outer_batch), replacement=True)
        xi_1 = self.sample_kernels(1, self.d_x)
        xi_2 = self.sample_kernels(self.Q, self.d_y)
        with torch.no_grad():
            self.log_y_2, log_H = self.get_log_H_all(self.model, xi_1, xi_2)
            self.log_y_2_prev, log_H_prev = self.get_log_H_all(self.model_prev, xi_1, xi_2)
            positive = torch.logaddexp(self.log_y_1 + math.log1p(-self.beta_k), log_H)
            negative = log_H_prev + math.log1p(-self.beta_k)
            difference = (negative - positive).clamp(max=0)
            self.log_y_1 = (positive + torch.log(-torch.expm1(difference))).clamp(
                self.log_y_1_min, self.log_y_1_max)
        self.model.zero_grad()
        log_H = self.batch_log_H(groups, xi_1, xi_2[None, :, :])
        log_scale = log_H.detach() - self.log_y_1[groups]
        offset = log_scale.max().clamp(min=0)
        (self.el * (log_H * torch.exp(log_scale - offset)).mean()).backward()
        with torch.no_grad():
            gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
            norm = torch.stack([g.norm() for g in gradients]).norm()
            if self.grad_clip is not None:
                offset = torch.minimum(offset, torch.log(norm.new_tensor(self.grad_clip)) - torch.log(norm))
            for gradient in gradients:
                gradient.copy_(gradient.sign() * torch.exp(torch.log(gradient.abs()) + offset))
        self.model_prev.load_state_dict(self.model.state_dict())
        if self.optimizer is not None:
            self.optimizer.step()
        else:
            with torch.no_grad():
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        parameter -= self.alpha_k * parameter.grad


class PortfolioWDROTrainer(CWDRO_Trainer):
    def train_rtmlmc(self, X_train, Y_train, pro_params):
        self.n_train, self.d_x = X_train.shape
        self.d_y, self.d_z = Y_train.shape[1], pro_params["dimension"][2]
        self.pro_params = pro_params
        self.lam = pro_params["lambda"]
        self.elements = np.arange(5) + 1
        probabilities = 0.5 ** (self.elements - 1)
        self.probabilities = probabilities / probabilities.sum()
        optimizer = torch.optim.Adam(self.model.parameters(), amsgrad=True,
                                     lr=self.params.LR_CWDRO)
        K = int(self.params.EPOCHS_SCSC)
        for step in range(1, K + 1):
            self.model.train()
            optimizer.zero_grad()
            losses = [self.RTMLMC_obj_oracle(X_train, Y_train, self.batch_size)
                      for _ in range(self.params.CWDRO_BATCH_SIZE)]
            torch.stack(losses).mean().backward()
            optimizer.step()
            if step == K:
                break
            if step % 5 == 0:
                self.model.eval()
                with torch.no_grad():
                    value = self.get_eval_loss(self.model, X_train, Y_train)
                self.train_loss_history.append(value.item())
        return self.model, self.train_loss_history


class Portfolio_Optimizer:
    def __init__(self, fixed_params, pro_params, best_parameters, device):
        self.params = copy.deepcopy(fixed_params)
        self.pro_params = pro_params
        self.best_parameters = best_parameters
        self.device = device
        self.d_x, self.d_y, self.d_z = pro_params['dimension']
        self.seed = pro_params['seed']
        self.p_norm_list = pro_params['p_norm_list']
        self.baseline_lr = pro_params['baseline_lr']

    def prepare_data(self):
        data = self.pro_params['data']
        date = self.pro_params['date']
        train_days = self.pro_params['train_days']
        stocks = np.random.RandomState(self.seed).choice(data.columns[:-self.d_x], self.d_y, replace=False)
        history = data.loc[data.index < date].tail(train_days)
        if len(history) != train_days:
            raise ValueError('Insufficient historical observations before the decision date.')
        features = history.iloc[:, -self.d_x:]
        mean, std = features.mean(), features.std() + 1e-8
        X = torch.tensor(((features - mean) / std).values, dtype=torch.float64, device=self.device)
        Y = torch.tensor(history[stocks].values, dtype=torch.float64, device=self.device)
        Xt = torch.tensor(((data.loc[date:].iloc[:1, -self.d_x:] - mean) / std).values,
                          dtype=torch.float64, device=self.device)
        future = data.loc[date:, stocks].head(60).values
        if len(future) != 60:
            raise ValueError('Portfolio evaluation requires 60 subsequent observations.')
        X, Y, groups = group_portfolio_data(X, Y, self.pro_params['n_m'])
        loader = DataLoader(TensorDataset(X, Y), batch_size=self.params.BATCH_SIZE, shuffle=False)
        q = {
            'problem_name': 'Portfolio',
            'eta': float(self.pro_params['eta']),
            'seed': self.seed,
            'historical_days': self.pro_params['historical_days'],
            'train_days': train_days,
            'n_m': self.pro_params['n_m'],
            'B': self.params.B,
            'a': self.params.a,
            'z0_bound': self.params.z0_bound,
            'dimension': [self.d_x, self.d_y, self.d_z],
            'group_index': groups,
            'future_return': future
        }
        return X, Y, Xt, future, loader, q


    def parameter_row(self, table, name, eta, norm, n_m):
        rows = table[(table['Model'] == name) & (table['ETA'] == eta) &
                     (table['norm'] == norm) & (table['n_m'] == n_m)]
        if len(rows) != 1:
            raise ValueError(f'Expected one parameter row for {(name, eta, norm, n_m)}.')
        return rows.iloc[0].copy()


    def make_model(self, params, model_type='SRF'):
        portfolio_seed(self.seed)
        if model_type == 'TNN':
            return TNN(self.d_x, self.params.hidden_layer_dim, self.d_z).to(self.device).double()
        return SRF(self.d_x, self.params.SRF_depth, self.d_z, self.params.Tree_number, params).to(self.device).double()


    def model_returns(self, model, X_today, future):
        model.eval()
        with torch.no_grad():
            weights, _ = parse_mv_output(model(X_today), self.d_z)
        return future @ weights[0].detach().cpu().numpy()


    def return_loss(self, returns, eta):
        # Compute only F; DRO baselines do not need Sharpe, CVaR or Std.
        return float(np.var(returns, ddof=0) - eta * np.mean(returns))


    def markowitz_weights(self, returns, eta, post_hoc=False):
        mean = returns.mean(axis=0)
        weights = cp.Variable(returns.shape[1])
        if post_hoc:
            risk = cp.sum_squares((returns - mean) @ weights) / len(returns)
        else:
            cov = np.cov(returns, rowvar=False) + 1e-6 * np.eye(returns.shape[1])
            risk = cp.quad_form(weights, cov)
        problem = cp.Problem(cp.Minimize(risk - eta * (mean @ weights)),
                             [weights >= 0, cp.sum(weights) == 1])
        problem.solve(solver='CLARABEL')
        if problem.status not in ('optimal', 'optimal_inaccurate'):
            raise RuntimeError('Portfolio quadratic optimization failed.')
        return weights.value


    def fit_erm(self, case, params, K, model_type='SRF'):
        X, Y, _, _, loader, q = case
        settings = copy.copy(params)
        settings.EPOCHS_SAA = math.ceil(K / len(loader))
        settings.LR_SAA = self.baseline_lr
        trainer = ERM_Trainer(self.make_model(settings, model_type), loader, settings, self.device, model_type)
        model, _ = trainer.train_erm(X, Y, q, progress=False,
                                     checkpoint_callback=lambda model, step: step < K,
                                     checkpoint_every=K)
        return model


    def fit_smooth_pair(self, case, params, smooth_row, row):
        # If epsilon agrees, save the warm start along the baseline training path.
        # This retains the same updates and random draws as two separate fits.
        X, Y, _, _, loader, q = case

        def fit(selected, steps):
            settings = copy.copy(params)
            K = max(steps)
            settings.EPOCHS_SAA = math.ceil(K / len(loader))
            settings.LR_SAA = selected['smooth_lr']
            pro_params = dict(q, norm=int(selected['norm']),
                              epsilon=float(selected['best_epsilon']),
                              **{'lambda': float(selected['best_lambda'])})
            trainer = Smooth_Trainer(self.make_model(settings), loader, settings, self.device, 'SRF')
            models = {}

            def record(model, step):
                if step in steps:
                    models[step] = copy.deepcopy(model)
                return step < K

            trainer.train_smooth(X, Y, pro_params, progress=False,
                                 checkpoint_callback=record, checkpoint_every=math.gcd(*steps))
            return models

        baseline_K, initial_K = int(smooth_row['K']), int(row['Smooth_K'])
        if float(smooth_row['best_epsilon']) == float(row['best_epsilon']):
            models = fit(smooth_row, sorted(set([baseline_K, initial_K])))
            return models[baseline_K], models[initial_K]
        baseline = fit(smooth_row, [baseline_K])[baseline_K]
        initial = fit(row, [initial_K])[initial_K]
        return baseline, initial


    def refine_smooth(self, initial, case, params, row, causal=True):
        X, Y, _, _, loader, q = case
        K = int(row['K'])
        if K == 0:
            return copy.deepcopy(initial)
        settings = copy.copy(params)
        settings.EPOCHS_SCSC = math.ceil(K / len(loader))
        settings.setLearningRate(row['cs_lr'], row['beta'])
        settings.Q = int(row['Q'])
        settings.outer_batch = int(row['Group_Batch'])
        settings.use_adam = True
        settings.grad_clip = 4.0
        settings.adam_decay = 0.0
        settings.structure_lr_scale = 1.0
        settings.warm_start_tracking_samples = 16
        q = dict(q, norm=int(row['norm']), epsilon=float(row['best_epsilon']),
                 causal_groups=causal, **{'lambda': float(row['best_lambda'])})
        portfolio_seed(self.seed + 100000)
        trainer = PortfolioTrackingBatch(copy.deepcopy(initial), loader, settings, self.device, 'SRF')
        model, _ = trainer.train_scsc(X, Y, q, progress=False,
                                      checkpoint_callback=lambda model, step: step < K,
                                      checkpoint_every=K)
        return model


    def fit_kl(self, case, params, row):
        X, Y, _, _, loader, q = case
        settings = copy.copy(params)
        settings.LR_KLDRO = self.baseline_lr
        trainer = KLDRO_Trainer(self.make_model(settings), loader, settings, self.device, 'SRF')
        trainer.pro_params = dict(q, **{'lambda': float(row['best_lambda'])})
        trainer.lam = float(row['best_lambda'])
        iterator = iter(loader)
        for _ in range(int(row['K'])):
            try:
                x_batch, y_batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                x_batch, y_batch = next(iterator)
            trainer.train_step(x_batch, y_batch)
        return trainer.model


    def financial_statistics(self, returns, eta, prefix):
        values = metrics(returns, eta)
        return {f'{name}_{prefix}': float(value)
                for name, value in zip(['loss', 'sharpe', 'cvar', 'std', 'mean'], values)}


    def fit_wdro(self, case, params, row, initial):
        X, Y, _, _, loader, q = case
        settings = copy.copy(params)
        K = int(row['K'])
        settings.EPOCHS_SCSC = K
        settings.LR_CWDRO = float(row['lr'])
        settings.CWDRO_BATCH_SIZE = int(row['Group_Batch'])
        settings.LR_CWDRO_INNER = float(row['inner_lr'])
        q = dict(q, norm=int(row['norm']), epsilon=0.0, **{'lambda': float(row['best_lambda'])})
        portfolio_seed(self.seed + 200000)
        trainer = PortfolioWDROTrainer(copy.deepcopy(initial), loader, settings, self.device, 'SRF')
        model, _ = trainer.train_rtmlmc(X, Y, q)
        return model


    def solve_all_models(self):
        date = self.pro_params['date']
        eta = self.pro_params['eta']
        n_m = self.pro_params['n_m']
        params = self.params
        table = self.best_parameters
        case = self.prepare_data()
        X, Y, Xt, future, loader, q = case
        statistics = dict(date=date, eta=eta, n_m=n_m, N=len(X))
        erm_model = None

        def get_erm_model():
            nonlocal erm_model
            if erm_model is None:
                erm_model = self.fit_erm(case, params, int(self.parameter_row(table, 'ERM', eta, 0, n_m)['K']))
            return erm_model

        def evaluate(column, compute_returns, prefix=None):
            returns = compute_returns()
            if not np.isfinite(returns).all():
                raise FloatingPointError(f'Nonfinite returns for {(date, eta, n_m, column)}.')
            F = self.return_loss(returns, eta)
            if prefix:
                statistics.update(self.financial_statistics(returns, eta, prefix))
            return F

        F_star = evaluate('F*', lambda: future @ self.markowitz_weights(future, eta, post_hoc=True), 'ph')
        F_erm = evaluate('F_ERM', lambda: self.model_returns(get_erm_model(), Xt, future), 'cmv_srf_saa')
        losses = {
            'KLDRO': evaluate('F_KLDRO', lambda: self.model_returns(
                self.fit_kl(case, params, self.parameter_row(table, 'KL-DRO', eta, 0, n_m)), Xt, future)),
            'MV': evaluate('F_MV', lambda: future @ self.markowitz_weights(Y.detach().cpu().numpy(), eta), 'nmv'),
            'EW': evaluate('F_EW', lambda: future.mean(axis=1), 'eq'),
            'CMV_TNN': evaluate('F_CMV_TNN', lambda: self.model_returns(
                self.fit_erm(case, params, int(self.parameter_row(table, 'CMV-TNN', eta, 0, n_m)['K']),
                        'TNN'), Xt, future), 'cmv_tnn_saa')}

        for norm in self.p_norm_list:
            row = self.parameter_row(table, 'Causal-SDRO', eta, norm, n_m)
            losses[f'{norm}_CWDRO'] = evaluate(f'F_{norm}_CWDRO', lambda: self.model_returns(
                self.fit_wdro(case, params, self.parameter_row(table, 'Causal-WDRO', eta, norm, n_m),
                         get_erm_model()), Xt, future))
            fitted = {}

            def get_model(name):
                if name in fitted:
                    return fitted[name]
                if 'Smooth' not in fitted:
                    smooth_row = self.parameter_row(table, 'Smooth', eta, norm, n_m)
                    fitted['Smooth'], fitted['Initial'] = self.fit_smooth_pair(case, params, smooth_row, row)
                if name == 'CSDRO':
                    fitted[name] = self.refine_smooth(fitted['Initial'], case, params, row)
                elif name == 'SDRO':
                    fitted[name] = (get_model('CSDRO') if n_m == 1 else
                                    self.refine_smooth(fitted['Initial'], case, params, row, causal=False))
                return fitted[name]

            losses[f'{norm}_Smooth'] = evaluate(f'F_{norm}_Smooth',
                lambda: self.model_returns(get_model('Smooth'), Xt, future), f'{norm}_smooth_srf')
            losses[f'{norm}_CSDRO'] = evaluate(f'F_{norm}_CSDRO',
                lambda: self.model_returns(get_model('CSDRO'), Xt, future), f'{norm}_sdmv_srf_scsc')
            losses[f'{norm}_SDRO'] = (losses[f'{norm}_CSDRO'] if n_m == 1 else
                evaluate(f'F_{norm}_SDRO', lambda: self.model_returns(get_model('SDRO'), Xt, future)))

        result = dict(Model='SRF(Constant Leaf)', Parameters=eta, N=len(X), D_X=self.d_x,
                      n_m=n_m, M=len(X)//n_m, Instance=date, **{'F*': F_star, 'F_ERM': F_erm})
        gap = F_erm - F_star
        names = {'KLDRO': 'KL-DRO', 'MV': 'Mean-Variance', 'EW': 'Equal-Weight', 'CMV_TNN': 'CMV-TNN'}
        for norm in self.p_norm_list:
            names.update({f'{norm}_CSDRO': f'{norm}-Causal-SDRO', f'{norm}_SDRO': f'{norm}-SDRO',
                          f'{norm}_CWDRO': f'{norm}-Causal-WDRO', f'{norm}_Smooth': f'{norm}-Smooth'})
        for name, F in losses.items():
            if not np.isfinite(F):
                raise FloatingPointError(f'Nonfinite {name} result for {(date, eta, n_m)}.')
            result[f'F_{name}'] = F
            result[f'P_{names[name]}'] = 100 * (F_erm - F) / gap if gap > 1e-10 else np.nan
        return result, statistics

