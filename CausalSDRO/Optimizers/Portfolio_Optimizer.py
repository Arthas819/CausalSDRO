"""
    This file solves portfolio mv models.
"""

import cvxpy as cp
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from CausalSDRO.Tools.Decision_Rules import *

from CausalSDRO.Optimizers.ERM_Trainer import *
from CausalSDRO.Optimizers.CSDRO_Trainer_Revise import *


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
    # Input a numpy
    if not isinstance(future_return, np.ndarray):
        future_return = np.array(future_return)

    T, n = future_return.shape

    # The mean and cov in the following (60) days
    mu = np.mean(future_return, axis=0)
    Sigma = np.cov(future_return, rowvar=False) + 1e-6 * np.eye(n)

    # Optimize: Variables
    w_var = cp.Variable(n)
    # Objective
    objective = cp.Minimize(cp.quad_form(w_var, Sigma) - ETA * (mu @ w_var))
    # Strategy
    constraints = [
        cp.sum(w_var) == 1,
        w_var >= 0
    ]
    prob = cp.Problem(objective, constraints)

    # Avoid the w_opt undefined
    w_opt = np.ones(n) / n

    # Solver
    try:
        prob.solve(solver=cp.GUROBI)
        # prob.solve(solver=cp.OSQP)
        # print('Gurobi')
    except Exception as e:
        prob.solve()
    else:
        if prob.status != 'optimal':
            print(f"Optimization status: {prob.status}, using equal weights")
            w_opt = np.ones(n) / n
        else:
            w_opt = w_var.value

    return w_opt


def solve_all_models(date, x_features, y_ret_select, q_params, f_params, device):
    # Input: date, features, returns, parameters (for question and fixed)
    best_parameters = pd.read_excel("../Outputs/Outputs_Tuning/DRO_Newsvendor_Best_Parameters.xlsx")

    # Results
    records = []

    # History features
    x_historical_1 = x_features.loc[:date, :]
    x_historical_row = x_historical_1.iloc[-q_params['historical_days']:, :]
    # Normalization
    scaler_mean = x_historical_row.mean()
    scaler_std = x_historical_row.std()
    x_historical = (x_historical_row - scaler_mean) / (scaler_std + 1e-8)

    # Feature Today
    x_today_row = x_features.loc[date:, :].iloc[:1, :]
    # Normalization (using mean and std of training set)
    x_today = (x_today_row - scaler_mean) / (scaler_std + 1e-8)

    # History return
    y_historical_1 = y_ret_select.loc[:date, :]
    y_historical = y_historical_1.iloc[-q_params['historical_days']:, :]

    # Future return
    future_return = y_ret_select[date:].head(60)

    # Transform to Tensor
    x_historical_tensor = torch.tensor(x_historical.values, dtype=torch.float64).to(device)
    y_historical_tensor = torch.tensor(y_historical.values, dtype=torch.float64).to(device)
    x_today_tensor = torch.tensor(x_today.values, dtype=torch.float64).to(device)

    # parameters
    eta = q_params['eta']

    ## ===== Baselines =====
    # Equal Weight Method
    w_eq = np.full(50, 1 / 50)
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
    train_loader = DataLoader(train_dataset, batch_size=f_params.BATCH_SIZE, shuffle=True)

    # Solve the Conditional Stochastic MV by SAA and SCSC
    # Train CMV with TNN by SAA
    print('Training Conditional Stochastic MV by ERM')
    # Model -> float 64
    model_cmv_tnn_saa_obj = TNN(input_dim=d_x, hidden_dim=f_params.hidden_layer_dim, output_dim=d_z).to(device).double()
    # Train this model as ERM, return a trained model and a historical loss list for drawing convergence curve
    erm_trainer = ERM_Trainer(model_cmv_tnn_saa_obj, train_loader, f_params, device, 'TNN')
    model_cmv_tnn_saa, _ = erm_trainer.train_erm(x_historical_tensor, y_historical_tensor, q_params)

    # Get weight and feature return
    w_cmv_tnn_saa, beta = parse_mv_output(model_cmv_tnn_saa(x_today_tensor), d_z)
    w_cmv_tnn_saa = w_cmv_tnn_saa.detach().cpu().numpy()
    return_cmv_tnn_saa = (future_return * w_cmv_tnn_saa).sum(axis=1)

    # Train CMV with SRF by SAA
    model_cmv_srf_saa_obj = SRF(input_dim=d_x, depth=f_params.SRF_depth, output_dim=d_z,
                                    tree_number=f_params.Tree_number, params = f_params).to(device).double()

    erm_trainer = ERM_Trainer(model_cmv_srf_saa_obj, train_loader, f_params, device, 'SRF')
    model_cmv_srf_saa, _ = erm_trainer.train_erm(x_historical_tensor, y_historical_tensor, q_params)

    # Get weight and feature return
    w_cmv_srf_saa, beta = parse_mv_output(model_cmv_srf_saa(x_today_tensor), d_z)
    w_cmv_srf_saa = w_cmv_srf_saa.detach().cpu().numpy()
    return_cmv_srf_saa = (future_return * w_cmv_srf_saa).sum(axis=1)

    # Now, SCSC
    print('Training Conditional Stochastic MV by SCSC')
    # Train CMV with TNN by SCSC
    model_cmv_tnn_scsc_obj = TNN(input_dim=d_x, hidden_dim=f_params.hidden_layer_dim, output_dim=d_z).to(device)
    model_cmv_tnn_scsc_obj = model_cmv_tnn_scsc_obj.double()
    # SCSC Trainer

    # Train the DRO model by SCSC
    scsc_trainer = CSDRO_Trainer(model_cmv_tnn_scsc_obj, train_loader, f_params, device, 'TNN')
    # Return a trained model and a historical loss list
    model_cmv_tnn_scsc, _ = scsc_trainer.train_scsc(x_historical_tensor, y_historical_tensor, q_params)
    # Get weight and feature return
    w_cmv_tnn_scsc, beta = parse_mv_output(model_cmv_tnn_scsc(x_today_tensor), d_z)
    w_cmv_tnn_scsc = w_cmv_tnn_scsc.detach().cpu().numpy()
    return_cmv_tnn_scsc = (future_return * w_cmv_tnn_scsc).sum(axis=1)

    # Train CMV with SRF by SCSC
    model_cmv_srf_scsc_obj = SRF(input_dim=d_x, depth=f_params.SRF_depth, output_dim=d_z,
                                 tree_number=f_params.Tree_number, params = f_params).to(device).double()
    # Train the DRO model by SCSC
    scsc_trainer = CSDRO_Trainer(model_cmv_srf_scsc_obj, train_loader, f_params, device, 'SRF')
    # Return a trained model and a historical loss list
    model_cmv_srf_scsc, _ = scsc_trainer.train_scsc(x_historical_tensor, y_historical_tensor, q_params)

    # Get weight and feature return
    w_cmv_srf_scsc, beta = parse_mv_output(model_cmv_srf_scsc(x_today_tensor), d_z)
    w_cmv_srf_scsc = w_cmv_srf_scsc.detach().cpu().numpy()
    return_cmv_srf_scsc = (future_return * w_cmv_srf_scsc).sum(axis=1)


    ## ===== DRO models ======
    # For 1-CausalSDRO-MV (p=1), TNN &SRF, SAA & SCSC

    # Best parameter combination
    best_row = best_parameters[
        (best_parameters['Model'] == 'Causal-SDRO') & (best_parameters['norm'] == 1)]
    q_params['lambda'] = best_row.iloc[0]['best_lambda']
    q_params['epsilon'] = best_row.iloc[0]['best_epsilon']
    q_params['norm'] = 1
    # Train 1-CausalSDRO-MV with TNN by SAA
    # Model
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
    model_1_sdmv_tnn_scsc_obj = TNN(input_dim=d_x, hidden_dim=f_params.hidden_layer_dim, output_dim=d_z).to(device)
    model_1_sdmv_tnn_scsc_obj = model_1_sdmv_tnn_scsc_obj.double()
    # Train the DRO model by SCSC
    scsc_trainer = CSDRO_Trainer(model_1_sdmv_tnn_scsc_obj, train_loader, f_params, device, 'TNN')
    # Return a trained model and a historical loss list
    model_1_sdmv_tnn_scsc, _ = scsc_trainer.train_scsc(x_historical_tensor, y_historical_tensor, q_params)

    # Get weight and feature return
    w_1_sdmv_tnn_scsc, beta = parse_mv_output(model_1_sdmv_tnn_scsc(x_today_tensor), d_z)
    w_1_sdmv_tnn_scsc = w_1_sdmv_tnn_scsc.detach().cpu().numpy()
    return_1_sdmv_tnn_scsc = (future_return * w_1_sdmv_tnn_scsc).sum(axis=1)


    # Train 1-CausalSDRO-MV with SRF by SCSC
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
    # Train 2-CausalSDRO-MV with TNN by SAA
    # Model
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
    model_2_sdmv_tnn_scsc_obj = TNN(input_dim=d_x, hidden_dim=f_params.hidden_layer_dim, output_dim=d_z).to(device)
    model_2_sdmv_tnn_scsc_obj = model_2_sdmv_tnn_scsc_obj.double()
    # Train the DRO model by SCSC
    scsc_trainer = CSDRO_Trainer(model_2_sdmv_tnn_scsc_obj, train_loader, f_params, device, 'TNN')
    # Return a trained model and a historical loss list
    model_2_sdmv_tnn_scsc, _ = scsc_trainer.train_scsc(x_historical_tensor, y_historical_tensor, q_params)

    # Get weight and feature return
    w_2_sdmv_tnn_scsc, beta = parse_mv_output(model_2_sdmv_tnn_scsc(x_today_tensor), d_z)
    w_2_sdmv_tnn_scsc = w_2_sdmv_tnn_scsc.detach().cpu().numpy()
    return_2_sdmv_tnn_scsc = (future_return * w_2_sdmv_tnn_scsc).sum(axis=1)

    # Train 2-CausalSDRO-MV with SRF by SCSC
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