'''
   This file solves the Feature-based Newsvendor Problem
'''

import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import pandas as pd

from Optimizers.ERM_Trainer import *
from Optimizers.CSDRO_Trainer import *

from Tools.Data_Generator_Newsvendor import *
from Tools.Decision_Rules import *
from Tools.Functions import *
from Tools.Plotting import *
from Tools.Parameters import *

'''
    Changeable Parameters
'''
# Select decision rule(s)
USE_TwoLayerNN = True
USE_SRF = True

# Output figures
Convergence_figure = False
Distribution_figures = True

# TwoLayerNN hidden layer dimension
hidden_layer_dim = 1000
# SRF hyperparameters, d epth and tree number
SRF_depth = 5
Tree_number = 20
# ERM epochs and total TNN Causal-SDRO updates
EPOCHS_SAA = 300
TNN_ITERATIONS = 1200

# Training sample size
n_train_list = [20, 40, 80, 100]
# Feature dimension
d_x_list = [3, 5, 10]
# Number of demand observations per distinct covariate
n_m_list = [1, 5, 10]
# Demand and Decision dimensions
d_z = 1
d_y = 1
# Number of instances for each scale
n_instances = 10
# Testing sample size for each instance
n_test = 10000

# DRO parameters
p_norm_list = [1, 2]
best_parameters = pd.read_excel(os.path.join(PROJECT_DIR, "Outputs/Outputs_Tuning/DRO_Newsvendor_Best_Parameters.xlsx"))

# Newsvendor cost, holding (h) and stockout (b) cost
h_cost_list = [0.6]
b_cost_list = [1.0]
# Standard deviation of the uncertain demand
sigma = 1

'''
    Basic Settings
'''
# Choose SRF type
Use_Linear_Leaf = False # False maybe better
# Use feature subsets in each SRT?
Use_Diverse_SRT = True
# Algorithm for solving DRO
DRO_USE_SCSC = True
DRO_USE_SAA = False  # Here we solve DRO model only using SCSC

# Fixed random seed
set_seed(42)

# Fixed parameters
params = Parameters_News()
params.setEpochs(EPOCHS_SAA, TNN_ITERATIONS)
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory
OUTPUT_DIR = os.path.join(PROJECT_DIR, "Outputs/Outputs_CSDRO")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

file_name = "DRO_Newsvendor_all_results.xlsx"

'''
    Main Function
'''
if __name__ == "__main__":
    # Record all results and output as an excel
    all_results = []
    # Scale
    for n_m in n_m_list:
        for n_train in n_train_list:
            for d_x in d_x_list:
                # Set decision rule parameters
                if USE_TwoLayerNN == True:
                    # Number of neurons is much lager than d_x
                    hidden_layer_dim = d_x * 50

                if USE_SRF == True:
                    # For each SRTree, depth = ⌈log_2 (d_x)⌉ + 1, e.g., d_x = 10 -> depth = 5
                    SRF_depth = math.ceil(math.log2(d_x)) + 1

                params.setModelStrcuture(hidden_layer_dim, SRF_depth, Tree_number, Use_Linear_Leaf, Use_Diverse_SRT)

                # Set initial learning rates for SCSC
                # Total TNN update budget; epochs are derived from the actual loader below.
                K = TNN_ITERATIONS
                # For SCSC, alpha and beta is proportional to O(1/k^{-1/2})
                LR_alpha = 1 / np.sqrt(K + 1e-9)
                LR_beta = 2 / np.sqrt(K + 1e-9)
                params.setLearningRate(LR_alpha, LR_beta)

                # For different parameter values
                for h_val, b_val in zip(h_cost_list, b_cost_list):
                    h_cost_tensor = torch.tensor([h_val]).to(device, dtype=torch.float64)
                    b_cost_tensor = torch.tensor([b_val]).to(device, dtype=torch.float64)

                    for p_norm in p_norm_list:

                        # Best parameter combination
                        best_row = best_parameters[
                            (best_parameters['Model'] == 'Causal-SDRO') & (best_parameters['norm'] == p_norm)]
                        lambda_val = best_row.iloc[0]['best_lambda']
                        epsilon_val = best_row.iloc[0]['best_epsilon']

                        # Newsvendor parameters
                        news_params = {
                            'problem_name': 'Newsvendor',
                            'h_cost': h_cost_tensor,
                            'b_cost': b_cost_tensor,
                            'lambda': lambda_val,
                            'epsilon': epsilon_val,
                            'norm': p_norm,
                            'dimension': [d_x,d_y,d_z],
                            'sigma': sigma,
                            'n_m': n_m
                        }
                        # For each instance:
                        for instance in range(1, 1 + n_instances):
                            news_params['seed'] = 42 + instance - 1
                            set_seed(news_params['seed'])
                            print(f"--- n_m={n_m}, Instance {instance}/{n_instances} ---")

                            # Generate data
                            data_gen = DataGenerator_News(n_train, n_test, news_params, device)
                            # Get normalized data
                            X_train, Y_train, X_test, Y_test, train_loader = data_gen.get_data(params.BATCH_SIZE)
                            F_star = calculate_F_star_saa_news(data_gen, X_test, news_params)

                            # Set decision rules
                            models_to_run = []
                            if USE_TwoLayerNN: models_to_run.append('TNN')
                            if USE_SRF: models_to_run.append('SRF')
                            # Draw the following models
                            models_for_plot2 = {}
                            instance_results = []

                            # For each model
                            for model_type in models_to_run:
                                model_name = None
                                print(f"--- Model: {model_type} ---")
                                # Create models, use .double() to set torch.float64
                                if model_type == 'TNN':
                                    model_name = 'TNN'
                                    # Baseline, solved as an ERM model (without Robustness)
                                    model_erm_baseline = TNN(input_dim=d_x, hidden_dim=hidden_layer_dim,
                                                                 output_dim=d_z).to(device).double()
                                    # Solved by SCSC as a DRO model
                                    model_csdro = TNN(input_dim=d_x, hidden_dim=hidden_layer_dim,
                                                     output_dim=d_z).to(device).double()

                                else:
                                    if Use_Linear_Leaf == True:
                                        model_name = 'SRF(Linear Leaf)'
                                    else:
                                        model_name = 'SRF(Constant Leaf)'
                                    # SAA baseline, solved by SAA as an ERM model (without Robustness)
                                    model_erm_baseline = SRF(input_dim=d_x, depth=SRF_depth, output_dim=d_z,
                                                                 tree_number=Tree_number, params = params).to(device).double()
                                    # Solved by SCSC as a DRO model
                                    model_csdro = SRF(input_dim=d_x, depth=SRF_depth, output_dim=d_z,
                                                         tree_number=Tree_number, params = params).to(device).double()

                                # Train this model as ERM, return a trained model and a historical loss list for drawing convergence curve
                                erm_trainer = ERM_Trainer(model_erm_baseline, train_loader, params, device, model_type)
                                model_erm, history_erm = erm_trainer.train_erm(X_train, Y_train, news_params)

                                # Use the SRF learning-rate scales selected in CV.
                                if model_type == 'SRF':
                                    params.EPOCHS_SCSC = math.ceil(best_row.iloc[0]['K'] / len(train_loader))
                                    K = params.EPOCHS_SCSC * len(train_loader)
                                    alpha_scale = best_row.iloc[0].get('best_alpha_scale', 1.0)
                                    beta_scale = best_row.iloc[0].get('best_beta_scale', 2.0)
                                    params.setLearningRate(alpha_scale / np.sqrt(K), beta_scale / np.sqrt(K))
                                else:
                                    params.EPOCHS_SCSC = math.ceil(TNN_ITERATIONS / len(train_loader))
                                    K = params.EPOCHS_SCSC * len(train_loader)
                                    params.setLearningRate(1 / np.sqrt(K), 2 / np.sqrt(K))
                                set_seed(news_params['seed'])
                                # Train the DRO model by SCSC
                                csdro_trainer = CSDRO_Trainer(model_csdro, train_loader, params, device, model_type)
                                # Return a trained model and a historical loss list
                                model_csdro, history_scsc = csdro_trainer.train_scsc(X_train, Y_train, news_params)

                                # For these models, evaluate their average loss on testing dataset
                                F_erm = get_test_loss(model_erm, X_test, Y_test, news_params)
                                F_dro = get_test_loss(model_csdro, X_test, Y_test, news_params)

                                # Results
                                regret = F_dro - F_star

                                # --- Store Results ---
                                instance_results.append({
                                    'Norm': p_norm, 'h': h_val, 'b': b_val, 'lambda': lambda_val, 'epsilon': epsilon_val,
                                    'N': n_train, 'D_X': d_x, 'n_m': n_m, 'Instance': instance, 'Model': model_name,
                                    'F*': F_star, 'F_ERM': F_erm, 'F_DRO': F_dro,
                                    'Regret': regret
                                })
                                # Convergence curve figures
                                if instance == 1 and Convergence_figure == True:
                                    convergence_data = {
                                        'ERM': history_erm,
                                        'SCSC': history_scsc
                                    }
                                    plot_convergence_curve(convergence_data, output_dir=OUTPUT_DIR,img_name=f"conv_N{n_train}_D{d_x}_nm{n_m}_p{p_norm}_{model_type}")

                                models_for_plot2[f'{model_type}_ERM'] = model_erm_baseline
                                models_for_plot2[f'{model_type}_SCSC'] = model_csdro

                            # Use the better ERM loss as a common baseline for this instance.
                            if instance_results:
                                F_erm_common = min(result['F_ERM'] for result in instance_results)
                                for result in instance_results:
                                    result['F_ERM'] = F_erm_common
                                    erm_gap = F_erm_common - result['F*']
                                    result['Prescriptiveness'] = (max(-1, 1 - result['Regret'] / erm_gap) * 100
                                                                  if erm_gap > 1e-10 else np.nan)
                                    print(result['Model'], ": F_trained = ", result['F_DRO'], ", F_ERM = ",
                                          result['F_ERM'], ", Out-of-sample Performance: ", result['Prescriptiveness'])
                                all_results.extend(instance_results)

                            # Distribution fitting figures
                            if Distribution_figures == True:
                                os.makedirs(os.path.join(OUTPUT_DIR, f"nm{n_m}_instance{instance}"), exist_ok=True)
                                plot_fit_visualization_newsvendor(data_gen, X_train, Y_train, models_for_plot2, news_params, os.path.join(OUTPUT_DIR, f"nm{n_m}_instance{instance}"))

    # --- Save all results to Excel ---
    df_results = pd.DataFrame(all_results)

    if not df_results.empty:
        print("\nSaving results to Excel...")

        columns_ordered = [
            'Norm', 'h', 'b', 'lambda', 'epsilon', 'N', 'D_X', 'n_m', 'Instance', 'Model',
            'F*', 'F_ERM', 'F_DRO', 'Regret', 'Prescriptiveness'
        ]

        final_columns = [col for col in columns_ordered if col in df_results.columns]
        df_to_save = df_results[final_columns]

        # Output path
        excel_path = os.path.join(OUTPUT_DIR, file_name)

        try:
            # Save to Excel
            df_to_save.to_excel(excel_path, index=False, sheet_name='All_Results')
            print(f"Successfully saved all results to {excel_path}")
        except Exception as e:
            print(f"Warning: Failed to save results to Excel. Error: {e}")

    print("All tasks completed.")
    