
"""
    This file provides a cross validation precedure for inventory substitute problem.
"""

import copy
import pandas as pd
import seaborn as sns

from CausalSDRO.Tools.Data_Generator_Inventory import *
from CausalSDRO.Tools.Decision_Rules import *
from CausalSDRO.Tools.Plotting import *
from CausalSDRO.Tools.Parameters import *

from CausalSDRO.Optimizers.ERM_Trainer import *
from CausalSDRO.Optimizers.CSDRO_Trainer_Revise import *
from CausalSDRO.Optimizers.SDRO_Trainer import *
from CausalSDRO.Optimizers.CWDRO_Trainer import *
from CausalSDRO.Optimizers.KLDRO_Trainer import *

'''
    Choose DRO Models
'''
# Select models
cv_csdro = False
cv_sdro = False
cv_cwdro = True
cv_kldro = False
# Record results of each model
models = []

'''
    Changeable Parameters
'''
lambda_list  = np.linspace(0.5, 2, 4)
epsilon_list = np.linspace(0.1, 0.4, 4)

# DRO parameters
p_norm_list = [1, 2]

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
EPOCHS_SAA  = 300
EPOCHS_SCSC = 300
# Training sample size
n_train_list = [400]
# Feature dimension
d_x_list = [20]
# Demand and Decision dimensions
d_z = 3
d_y = 3
# Number of instances for each scale
n_instances = 2
# Testing sample size for each instance
n_test = 10000

# Costs
h_cost_list = [ [1, 0.7, 0.6] ]
b_cost_list = [ [1.8, 1.6, 1.2] ]
c_cost_list = [ [0,0,0] ]
s_cost_list = [ [[0, 1.7, 2], [float("inf"), 0, 1.5], [float("inf"), float("inf"), 0]] ]

'''
    Basic Settings
'''
# Fixed parameters
params = Parameters_Inv()
params.setEpochs(EPOCHS_SAA, EPOCHS_SCSC)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Output path
output_filename = "DRO_Inventory_Best_Parameters_plus.xlsx"
OUTPUT_DIR = "../Outputs/Outputs_Tuning"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if __name__ == "__main__":

    # Record all parameters for all DRO models
    results_csdro   = []
    results_sdro    = []
    results_cwdro   = []
    results_kldro   = []
    best_parameters = []

    # For this problem, we have calculated its optimal value by SAA.
    F_star = 3.29
    print(f"Theoretical F* : {F_star:.6f}")

    for p_norm in p_norm_list:

        for n_train in n_train_list:

            for d_x in d_x_list:

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

                # For different parameter values
                for h_val, b_val, c_val, s_val in zip(h_cost_list, b_cost_list, c_cost_list, s_cost_list):
                    h_cost_tensor = torch.tensor(h_val).to(device, dtype=torch.float64)
                    b_cost_tensor = torch.tensor(b_val).to(device, dtype=torch.float64)
                    c_cost_tensor = torch.tensor(c_val).to(device, dtype=torch.float64)
                    s_cost_tensor = torch.tensor(s_val).to(device, dtype=torch.float64)

                    for lambda_val in lambda_list:

                        # Solve Causal-WDRO
                        if cv_cwdro == True:
                            inv_params = {
                                'problem_name': 'Inventory',
                                'h_cost': h_cost_tensor,
                                'b_cost': b_cost_tensor,
                                'c_cost': c_cost_tensor,
                                's_cost': s_cost_tensor,
                                'lambda': lambda_val,
                                'epsilon': 0,
                                'norm': p_norm,
                                'dimension': [d_x,d_y,d_z]
                            }
                            for instance in range(1, 1 + n_instances):
                                print(f"--- Causal-WDRO Instance {instance}/{n_instances}, lambda={lambda_val} ---")
                                # Generate data
                                data_gen = DataGenerator_Inventory(n_train, n_test, inv_params, device)
                                # Get normalized data
                                X_train, Y_train, X_test, Y_test, train_loader = data_gen.get_data(params.BATCH_SIZE)
                                # Solve Causal-WDRO
                                cwdro_trainer = CWDRO_Trainer(copy.deepcopy(model_srf), train_loader, params, device, 'SRF')
                                model_cwdro, history_cwdro = cwdro_trainer.train_rtmlmc(X_train, Y_train, inv_params)
                                F_cwdro = get_test_loss(model_cwdro, X_test, Y_test, inv_params)
                                # Solve ERM
                                erm_trainer = ERM_Trainer(copy.deepcopy(model_srf), train_loader, params, device,'SRF')
                                model_erm, history_saa = erm_trainer.train_erm(X_train, Y_train, inv_params)
                                F_erm = get_test_loss(model_erm, X_test, Y_test, inv_params)
                                # Out-of-sample performance (%)
                                P_value = max(-1, 1 - (F_cwdro - F_star) / (F_erm - F_star)) * 100
                                # Record
                                print('CWDRO P:', P_value, ',   F_cwdro:', F_cwdro, ',   F_erm:', F_erm)
                                results_cwdro.append({'norm':p_norm, 'lambda': lambda_val, 'epsilon': 0, 'P': P_value})

                        if cv_kldro == True and p_norm == p_norm_list[0]: # Only solve KL-DRO once
                            inv_params = {
                                'problem_name': 'Inventory',
                                'h_cost': h_cost_tensor,
                                'b_cost': b_cost_tensor,
                                'c_cost': c_cost_tensor,
                                's_cost': s_cost_tensor,
                                'lambda': lambda_val,
                                'epsilon': 0,
                                'norm': p_norm,
                                'dimension': [d_x,d_y,d_z]
                            }
                            for instance in range(1, 1 + n_instances):
                                print(f"--- KL-DRO Instance {instance}/{n_instances}, lambda={lambda_val} ---")
                                # Generate data
                                data_gen = DataGenerator_Inventory(n_train, n_test, inv_params, device)
                                # Get normalized data
                                X_train, Y_train, X_test, Y_test, train_loader = data_gen.get_data(params.BATCH_SIZE)
                                # Solve KL-DRO
                                kldro_trainer = KLDRO_Trainer(copy.deepcopy(model_srf), train_loader, params, device, 'SRF')
                                model_kldro, history_kldro = kldro_trainer.train_sgd(X_train, Y_train, inv_params)
                                F_kldro = get_test_loss(model_kldro, X_test, Y_test, inv_params)
                                # Solve ERM
                                erm_trainer = ERM_Trainer(copy.deepcopy(model_srf), train_loader, params, device,'SRF')
                                model_erm, history_saa = erm_trainer.train_erm(X_train, Y_train, inv_params)
                                F_erm = get_test_loss(model_erm, X_test, Y_test, inv_params)
                                # Out-of-sample performance (%)
                                P_value = max(-1, 1 - (F_kldro - F_star) / (F_erm - F_star)) * 100
                                # Record
                                print('KLDRO P:', P_value, ',   F_kldro:', F_kldro, ',   F_erm:', F_erm)
                                results_kldro.append(
                                    {'norm': 0, 'lambda': lambda_val, 'epsilon': 0, 'P': P_value})

                        if cv_csdro == True or cv_sdro == True:
                            for epsilon_val in epsilon_list:
                                inv_params = {
                                    'problem_name': 'Inventory',
                                    'h_cost': h_cost_tensor,
                                    'b_cost': b_cost_tensor,
                                    'c_cost': c_cost_tensor,
                                    's_cost': s_cost_tensor,
                                    'lambda': lambda_val,
                                    'epsilon': epsilon_val,
                                    'norm': p_norm,
                                    'dimension': [d_x, d_y, d_z]
                                }
                                for instance in range(1, 1 + n_instances):
                                    print(f"--- Instance {instance}/{n_instances} , lambda={lambda_val}, epsilon={epsilon_val}---")
                                    # Generate data
                                    data_gen = DataGenerator_Inventory(n_train, n_test, inv_params, device)
                                    # Get normalized data
                                    X_train, Y_train, X_test, Y_test, train_loader = data_gen.get_data(params.BATCH_SIZE)
                                    # Solve ERM
                                    erm_trainer = ERM_Trainer(copy.deepcopy(model_srf), train_loader, params, device,'SRF')
                                    model_erm, history_saa = erm_trainer.train_erm(X_train, Y_train, inv_params)
                                    F_erm = get_test_loss(model_erm, X_test, Y_test, inv_params)
                                    # Solve DRO
                                    if cv_csdro == True:
                                        print(f"--- Causal-SDRO Instance {instance}/{n_instances} , lambda={lambda_val}, epsilon={epsilon_val}---")
                                        csdro_trainer = CSDRO_Trainer(copy.deepcopy(model_srf), train_loader, params, device, 'SRF')
                                        model_csdro, history_csdro = csdro_trainer.train_scsc(X_train, Y_train, inv_params)
                                        F_csdro = get_test_loss(model_csdro, X_test, Y_test, inv_params)
                                        # Out-of-sample performance (%)
                                        P_value = max(-1, 1 - (F_csdro - F_star) / (F_erm - F_star)) * 100
                                        # Record
                                        print('CSDRO P:', P_value, ',   F_csdro:', F_csdro, ',   F_erm:', F_erm)
                                        results_csdro.append(
                                            {'norm': p_norm, 'lambda': lambda_val, 'epsilon': epsilon_val, 'P': P_value})
                                    else:
                                        print(f"--- SDRO Instance {instance}/{n_instances} , lambda={lambda_val}, epsilon={epsilon_val}---")
                                        sdro_trainer = SDRO_Trainer(copy.deepcopy(model_srf), train_loader, params, device, 'SRF')
                                        model_sdro, history_sdro = sdro_trainer.train_scsc(X_train, Y_train, inv_params)
                                        F_sdro = get_test_loss(model_sdro, X_test, Y_test, inv_params)
                                        # Out-of-sample performance (%)
                                        P_value = max(-1, 1 - (F_sdro - F_star) / (F_erm - F_star)) * 100
                                        # Record
                                        print('SDRO P:', P_value, ',   F_sdro:', F_sdro, ',   F_erm:', F_erm)
                                        results_sdro.append(
                                            {'norm': p_norm, 'lambda': lambda_val, 'epsilon': epsilon_val, 'P': P_value})

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
        df_dro_aver = df_dro.groupby(['norm', 'lambda', 'epsilon'])['P'].mean().reset_index()
        # For p-Causal-SDRO, generate heatmap and select best combination
        unique_norms = df_dro_aver['norm'].unique()
        for current_norm in unique_norms:
            print(f"Plotting heatmap for norm = {current_norm}...")
            # Select specific norm value
            df_subset = df_dro_aver[df_dro_aver['norm'] == current_norm]
            # Find the best parameter combination
            best_idx = df_subset['P'].idxmax()
            best_row = df_subset.loc[best_idx]
            best_parameters.append(
                {'Model': model_name, 'norm': current_norm,
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
            plt.title(f'{current_norm}-{model_name} with n={n_train_list[0]}')
            plt.tight_layout()
            filename_base = f"Inventory-{model_name}-{current_norm}_Heatmap"
            plt.savefig(os.path.join(OUTPUT_DIR, f"{filename_base}.png"), format='png')
            plt.savefig(os.path.join(OUTPUT_DIR, f"{filename_base}.pdf"), format='pdf')
            plt.close()

    # Save best parameters
    df_params = pd.DataFrame(best_parameters)

    file_path = os.path.join(OUTPUT_DIR, output_filename)
    df_params.to_excel(file_path, index=False)