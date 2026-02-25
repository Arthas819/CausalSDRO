
"""
    This file provides a cross validation precedure for portfolio.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import copy
import seaborn as sns

from CausalSDRO.Tools.Plotting import *
from CausalSDRO.Tools.Parameters import *

from CausalSDRO.Optimizers.SDRO_Trainer import *
from CausalSDRO.Optimizers.CWDRO_Trainer import *
from CausalSDRO.Optimizers.KLDRO_Trainer import *
from CausalSDRO.Optimizers.Portfolio_Optimizer import *

'''
    Choose DRO Models
'''
# Select models
cv_csdro = True
cv_sdro = False
cv_cwdro = False
cv_kldro = False
# Record results of each model
models = []

'''
    Choose trade dates for decision
'''

set_trade_dates = ['2021-01-03', '2021-04-01', '2021-07-01', '2021-10-01',
                   '2022-01-03', '2022-04-01', '2022-07-01', '2022-10-03']

'''
    Changeable Parameters
'''
lambda_list  = np.linspace(0.5, 1.1, 4)
epsilon_list = np.linspace(0.5, 1.1, 4)

# DRO parameter
p_norm_list = [1, 2]
# Portfolio parameter
ETA_List = [7]

# Decision rule
USE_SRF = True
# Choose SRF type
Use_Linear_Leaf = False
# Use feature subsets in each SRT?
Use_Diverse_SRT = False
# TwoLayerNN hidden layer dimension
hidden_layer_dim = 512
# SRF hyperparameters, depth and tree number
SRF_depth = 5
Tree_number = 20
# Epochs for SAA and SCSC
EPOCHS_SAA  = 400
EPOCHS_SCSC = 400
# Training sample size
n_train = HISTORY_DAYS = 730
# Feature, demand, and decision dimensions
STOCK_NUM = 50
d_x = 5
d_y = STOCK_NUM
d_z = STOCK_NUM + 1

RANDOM_SEED = 42

'''
    Basic Settings
'''

# Fixed parameters
params = Parameters_Portfolio()
params.setEpochs(EPOCHS_SAA, EPOCHS_SCSC)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Output path
output_filename = "DRO_Portfolio_Best_Parameters_plus.xlsx"
OUTPUT_DIR = "../Outputs/Outputs_Tuning"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

'''
    Read Data
'''

all_data = pd.read_csv('data/[Return and Features]20170101-20230331.csv')
all_data = all_data.set_index(all_data.columns[0])
x_features = all_data.iloc[:, -5:]
y_returns  = all_data.iloc[:, :-5]

if __name__ == "__main__":

    # Select Stocks Randomly
    rng = np.random.RandomState(RANDOM_SEED)
    selected_cols = rng.choice(y_returns.columns, size=STOCK_NUM, replace=False)
    y_ret_select = y_returns[selected_cols]

    # Record all parameters for all DRO models
    results_csdro   = []
    results_sdro    = []
    results_cwdro   = []
    results_kldro   = []
    best_parameters = []

    for p_norm in p_norm_list:

        for ETA in ETA_List:

            SRF_depth = math.ceil(math.log2(d_x)) + 1
            params.setModelStrcuture(hidden_layer_dim, SRF_depth, Tree_number, Use_Linear_Leaf, Use_Diverse_SRT)

            # For SCSC, alpha and beta is proportional to O(1/k^{-1/2})
            K = EPOCHS_SCSC * (n_train // params.BATCH_SIZE)  # Total iteration times, train a batch-size
            LR_alpha = 1 / np.sqrt(K + 1e-9)
            LR_beta = 2 / np.sqrt(K + 1e-9)
            params.setLearningRate(LR_alpha, LR_beta)

            # SRF model
            model_srf = SRF(input_dim=d_x, depth=SRF_depth, output_dim=d_z,
                            tree_number=Tree_number, params=params).to(device).double()

            for date in set_trade_dates:

                # Transform data to training and testing sets

                # History features
                x_historical_1 = x_features.loc[:date, :]
                x_historical_row = x_historical_1.iloc[-HISTORY_DAYS:, :]
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
                y_historical = y_historical_1.iloc[-HISTORY_DAYS:, :]
                # Return today
                y_today = y_ret_select.loc[date:, :].iloc[:1, :]
                # Future return
                future_return = y_ret_select[date:].head(60)

                # Transform to Tensor. Training and testing sets
                X_train = torch.tensor(x_historical.values, dtype=torch.float64).to(device)
                Y_train = torch.tensor(y_historical.values, dtype=torch.float64).to(device)
                X_test = torch.tensor(x_today.values, dtype=torch.float64).to(device)
                Y_test = torch.tensor(y_today.values, dtype=torch.float64).to(device)
                # Set train loader
                train_dataset = TensorDataset(X_train, Y_train)
                train_loader = DataLoader(
                    dataset = train_dataset,
                    batch_size = params.BATCH_SIZE,
                    shuffle = False
                )

                # Baseline, Post-hoc (Complete Information) Method
                weight_post_hc = post_hc_solver(future_return.values, ETA)
                return_post_hc = (future_return * weight_post_hc).sum(axis=1)
                F_star, _, _, _, _ = metrics(return_post_hc, ETA)

                for lambda_val in lambda_list:

                    # Solve Causal-WDRO
                    if cv_cwdro == True:
                        pf_params = {
                            'problem_name': 'Portfolio',
                            'eta': ETA,
                            'historical_days': HISTORY_DAYS,
                            'lambda': lambda_val,
                            'epsilon': 0,
                            'norm': p_norm,
                            'dimension': [d_x, d_y, d_z],
                            'future_return': future_return
                        }
                        print(f"--- Causal-WDRO Date: {date}, lambda={lambda_val} ---")
                        # Solve Causal-WDRO
                        cwdro_trainer = CWDRO_Trainer(copy.deepcopy(model_srf), train_loader, params, device, 'SRF')
                        model_cwdro, history_cwdro = cwdro_trainer.train_rtmlmc(X_train, Y_train, pf_params)
                        F_cwdro = get_test_loss(model_cwdro, X_test, Y_test, pf_params)
                        # Solve ERM
                        erm_trainer = ERM_Trainer(copy.deepcopy(model_srf), train_loader, params, device,'SRF')
                        model_erm, history_saa = erm_trainer.train_erm(X_train, Y_train, pf_params)
                        F_erm = get_test_loss(model_erm, X_test, Y_test, pf_params)
                        # Out-of-sample performance (%)
                        P_value = max(-1, 1 - (F_cwdro - F_star) / (F_erm - F_star)) * 100
                        # Record
                        print('CWDRO P:', P_value, ',   F*:', F_star, ',   F_cwdro:', F_cwdro, ',   F_erm:', F_erm)
                        results_cwdro.append({'norm':p_norm, 'ETA':ETA, 'lambda': lambda_val, 'epsilon': 0, 'P': P_value})

                    if cv_kldro == True and p_norm == p_norm_list[0]: # Only solve KL-DRO once
                        pf_params = {
                            'problem_name': 'Portfolio',
                            'eta': ETA,
                            'historical_days': HISTORY_DAYS,
                            'lambda': lambda_val,
                            'epsilon': 0,
                            'norm': p_norm,
                            'dimension': [d_x, d_y, d_z],
                            'future_return': future_return
                        }
                        print(f"--- KL-DRO Date: {date}, lambda={lambda_val} ---")
                        # Solve KL-DRO
                        kldro_trainer = KLDRO_Trainer(copy.deepcopy(model_srf), train_loader, params, device, 'SRF')
                        model_kldro, history_kldro = kldro_trainer.train_sgd(X_train, Y_train, pf_params)
                        F_kldro = get_test_loss(model_kldro, X_test, Y_test, pf_params)
                        # Solve ERM
                        erm_trainer = ERM_Trainer(copy.deepcopy(model_srf), train_loader, params, device,'SRF')
                        model_erm, history_saa = erm_trainer.train_erm(X_train, Y_train, pf_params)
                        F_erm = get_test_loss(model_erm, X_test, Y_test, pf_params)
                        # Out-of-sample performance (%)
                        P_value = max(-1, 1 - (F_kldro - F_star) / (F_erm - F_star)) * 100
                        # Record
                        print('KLDRO P:', P_value, ',   F*:', F_star, ',   F_kldro:', F_kldro, ',   F_erm:', F_erm)
                        results_kldro.append(
                            {'norm': 0, 'ETA':ETA, 'lambda': lambda_val, 'epsilon': 0, 'P': P_value})

                    if cv_csdro == True or cv_sdro == True:
                        for epsilon_val in epsilon_list:
                            # Update parameters
                            pf_params = {
                                'problem_name': 'Portfolio',
                                'eta': ETA,
                                'historical_days': HISTORY_DAYS,
                                'lambda': lambda_val,
                                'epsilon': epsilon_val,
                                'norm': p_norm,
                                'dimension': [d_x, d_y, d_z],
                                'future_return': future_return
                            }
                            # Solve ERM
                            erm_trainer = ERM_Trainer(copy.deepcopy(model_srf), train_loader, params, device,'SRF')
                            model_erm, history_saa = erm_trainer.train_erm(X_train, Y_train, pf_params)
                            F_erm = get_test_loss(model_erm, X_test, Y_test, pf_params)
                            # Solve DRO
                            if cv_csdro == True:
                                print(f"--- CSDRO Date: {date}, lambda={lambda_val}, epsilon={epsilon_val}---")
                                csdro_trainer = CSDRO_Trainer(copy.deepcopy(model_srf), train_loader, params, device, 'SRF')
                                model_csdro, history_csdro = csdro_trainer.train_scsc(X_train, Y_train, pf_params)
                                F_csdro = get_test_loss(model_csdro, X_test, Y_test, pf_params)
                                # Out-of-sample performance (%)
                                P_value = max(-1, 1 - (F_csdro - F_star) / (F_erm - F_star)) * 100
                                # Record
                                print('CSDRO P:', P_value, ',   F*:', F_star, ',   F_csdro:', F_csdro, ',   F_erm:', F_erm)
                                results_csdro.append(
                                    {'norm': p_norm, 'ETA':ETA, 'lambda': lambda_val, 'epsilon': epsilon_val, 'P': P_value})
                            else:
                                print(f"--- SDRO Date: {date}, lambda={lambda_val}, epsilon={epsilon_val}---")
                                sdro_trainer = SDRO_Trainer(copy.deepcopy(model_srf), train_loader, params, device, 'SRF')
                                model_sdro, history_sdro = sdro_trainer.train_scsc(X_train, Y_train, pf_params)
                                F_sdro = get_test_loss(model_sdro, X_test, Y_test, pf_params)
                                # Out-of-sample performance (%)
                                P_value = max(-1, 1 - (F_sdro - F_star) / (F_erm - F_star)) * 100
                                # Record
                                print('SDRO P:', P_value, ',   F*:', F_star, ',   F_sdro:', F_sdro, ',   F_erm:', F_erm)
                                results_sdro.append(
                                    {'norm': p_norm, 'ETA':ETA, 'lambda': lambda_val, 'epsilon': epsilon_val, 'P': P_value})

    # Save results as a tuple
    if cv_csdro == True:
        models.append(('Causal-SDRO', results_csdro))
    if cv_sdro == True:
        models.append(('SDRO', results_sdro))
    if cv_cwdro == True:
        models.append(('Causal-WDRO', results_cwdro))
    if cv_kldro == True:
        models.append(('KL-DRO', results_kldro))

    # Summarize Results
    for model_name, results_table in models:
        print(model_name)
        df_dro = pd.DataFrame(results_table)
        # Average over instances
        df_dro_aver = df_dro.groupby(['norm', 'ETA', 'lambda', 'epsilon'])['P'].mean().reset_index()
        # For p-Causal-SDRO, generate heatmap and select best combination
        unique_norms = df_dro_aver['norm'].unique()
        for current_norm in unique_norms:
            for ETA in ETA_List:
                print(f"Plotting heatmap for norm = {current_norm}...")
                # Select specific norm value
                df_subset = df_dro_aver[df_dro_aver['norm'] == current_norm]
                df_subset = df_subset[df_subset['ETA'] == ETA]
                # Find the best parameter combination
                best_idx = df_subset['P'].idxmax()
                best_row = df_subset.loc[best_idx]
                best_parameters.append(
                    {'Model': model_name, 'norm': current_norm, 'ETA': ETA,
                     'best_lambda': best_row['lambda'], 'best_epsilon': best_row['epsilon'], 'best_P': best_row['P']})
                # Heatmap Preparation
                pivot = df_subset.pivot(index='lambda', columns='epsilon', values='P')
                pivot = pivot.sort_index(ascending=True).sort_index(axis=1, ascending=True)
                # Draw
                plt.figure(figsize=(10, 8))
                x_lbl = [f"{x:.1e}" for x in pivot.columns]
                y_lbl = [f"{y:.1e}" for y in pivot.index]
                ax = sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                                 xticklabels=x_lbl, yticklabels=y_lbl,
                                 mask=pivot.isnull(),
                                 cbar_kws={'label': 'Out-of-sample Performance'})
                ax.invert_yaxis()
                plt.xlabel('Regularization Parameter $\epsilon$')
                plt.ylabel('Penalty Parameter $\lambda$')
                plt.title(f'{current_norm}-{model_name} with n={HISTORY_DAYS}')
                plt.tight_layout()
                filename_base = f"Portfolio-{model_name}-{current_norm}_Heatmap"
                plt.savefig(os.path.join(OUTPUT_DIR, f"{filename_base}.png"), format='png')
                plt.savefig(os.path.join(OUTPUT_DIR, f"{filename_base}.pdf"), format='pdf')
                plt.close()

    # Save best parameters
    df_params = pd.DataFrame(best_parameters)

    file_path = os.path.join(OUTPUT_DIR, output_filename)
    df_params.to_excel(file_path, index=False)