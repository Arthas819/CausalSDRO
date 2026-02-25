'''
   This file solves the Feature-based Inventory Institute Problem
'''

import pandas as pd

from CausalSDRO.Optimizers.ERM_Trainer import *
from CausalSDRO.Optimizers.CSDRO_Trainer_Revise import *

from CausalSDRO.Tools.Data_Generator_Inventory import *
from CausalSDRO.Tools.Decision_Rules import *
from CausalSDRO.Tools.Functions import *
from CausalSDRO.Tools.Plotting import *
from CausalSDRO.Tools.Parameters import *

'''
    Changeable Parameters
'''
# Select decision rule(s)
USE_TwoLayerNN = True
USE_SRF = True

# Output figures
Convergence_figure = False

# TwoLayerNN hidden layer dimension
hidden_layer_dim = 1000
# SRF hyperparameters, d epth and tree number
SRF_depth = 5
Tree_number = 20
# Epochs for SAA and SCSC
EPOCHS_SAA  = 300
EPOCHS_SCSC = 300

# Training sample size
n_train_list = [100, 200, 400, 800]
# Feature dimension
d_x_list = [3, 5, 10, 20]
# Demand and Decision dimensions
d_z = 3
d_y = 3
# Number of instances for each scale
n_instances = 10
# Testing sample size for each instance
n_test = 10000

# DRO parameters
p_norm_list = [1, 2]
best_parameters = pd.read_excel("../Outputs/Outputs_Tuning/DRO_Inventory_Best_Parameters.xlsx")

# Inventory cost, holding (h), stockout (b), purchase (c), and substitution (S) cost
h_cost_list = [ [1, 0.7, 0.6] ]
b_cost_list = [ [1.8, 1.6, 1.2] ]
c_cost_list = [ [0,0,0] ]
s_cost_list = [ [[0, 1.7, 2], [float("inf"), 0, 1.5], [float("inf"), float("inf"), 0]] ]


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

# Fixed parameters
params = Parameters_Inv()
params.setEpochs(EPOCHS_SAA, EPOCHS_SCSC)
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory
OUTPUT_DIR = "../Outputs/Outputs_CSDRO"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

file_name = "DRO_Inventory_all_results.xlsx"

'''
    Main Function
'''
if __name__ == "__main__":
    # Record all results and output as an excel
    all_results = []
    # Scale
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
            # In each epoch, the parameters are updated (n_train / batch_size) times, and thus the total K is
            K = EPOCHS_SCSC * (n_train // params.BATCH_SIZE)
            # For SCSC, alpha and beta is proportional to O(1/k^{-1/2})
            LR_alpha = 1 / np.sqrt(K + 1e-9)
            LR_beta = 2 / np.sqrt(K + 1e-9)
            params.setLearningRate(LR_alpha, LR_beta)

            # For different parameter values
            for h_val, b_val, c_val, s_val in zip(h_cost_list, b_cost_list, c_cost_list, s_cost_list):
                h_cost_tensor = torch.tensor(h_val).to(device, dtype=torch.float64)
                b_cost_tensor = torch.tensor(b_val).to(device, dtype=torch.float64)
                c_cost_tensor = torch.tensor(c_val).to(device, dtype=torch.float64)
                s_cost_tensor = torch.tensor(s_val).to(device, dtype=torch.float64)

                for p_norm in p_norm_list:

                    # Best parameter combination
                    best_row = best_parameters[
                        (best_parameters['Model'] == 'Causal-SDRO') & (best_parameters['norm'] == p_norm)]
                    lambda_val = best_row.iloc[0]['best_lambda']
                    epsilon_val = best_row.iloc[0]['best_epsilon']

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

                    # For each instance:
                    for instance in range(1, 1 + n_instances):
                        print(f"--- Instance {instance}/{n_instances} ---")

                        # Generate data
                        data_gen = DataGenerator_Inventory(n_train, n_test, inv_params, device)
                        # Get normalized data
                        X_train, Y_train, X_test, Y_test, train_loader = data_gen.get_data(params.BATCH_SIZE)

                        # Calculate theoretical F*
                        F_star = calculate_F_star_theoretical_is(data_gen, X_test, Y_test, inv_params)
                        print(f"Theoretical F* : {F_star:.6f}")

                        # Set decision rules
                        models_to_run = []
                        if USE_TwoLayerNN: models_to_run.append('TNN')
                        if USE_SRF: models_to_run.append('SRF')
                        # Draw the following models
                        models_for_plot2 = {}

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
                            model_erm, history_erm = erm_trainer.train_erm(X_train, Y_train, inv_params)

                            # Train the DRO model by SCSC
                            csdro_trainer = CSDRO_Trainer(model_csdro, train_loader, params, device, model_type)
                            # Return a trained model and a historical loss list
                            model_csdro, history_scsc = csdro_trainer.train_scsc(X_train, Y_train, inv_params)

                            # For these models, evaluate their average loss on testing dataset
                            F_erm = get_test_loss(model_erm, X_test, Y_test, inv_params)
                            F_dro = get_test_loss(model_csdro, X_test, Y_test, inv_params)

                            # Results
                            regret = F_dro - F_star
                            erm_gap = F_erm - F_star
                            # Out-of-sample performance
                            prescriptiveness = max(-1, 1 - regret / erm_gap) * 100

                            print(model_name, ": F_trained = ", F_dro, ", F_ERM = ", F_erm, ", Out-of-sample Performance: " , prescriptiveness)

                            # --- Store Results ---
                            all_results.append({
                                'Norm': p_norm, 'lambda': lambda_val, 'epsilon': epsilon_val,
                                'N': n_train, 'D_X': d_x, 'Instance': instance, 'Model': model_name,
                                'F*': F_star, 'F_ERM': F_erm, 'F_DRO': F_dro,
                                'Regret': regret, 'Prescriptiveness': prescriptiveness
                            })
                            # Convergence curve figures
                            if instance == 1 and Convergence_figure == True:
                                convergence_data = {
                                    'ERM': history_erm,
                                    'SCSC': history_scsc
                                }
                                plot_convergence_curve(convergence_data, output_dir=OUTPUT_DIR,
                                                       img_name=f"conv_N{n_train}_D{d_x}_{model_type}")

                            models_for_plot2[f'{model_type}_ERM'] = model_erm
                            models_for_plot2[f'{model_type}_SCSC'] = model_csdro

    # --- Save all results to Excel ---
    df_results = pd.DataFrame(all_results)

    if not df_results.empty:
        print("\nSaving results to Excel...")

        columns_ordered = [
            'Norm', 'lambda', 'epsilon', 'N', 'D_X', 'Instance', 'Model',
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