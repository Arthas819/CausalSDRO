'''
   This file compares Causal-SDRO, SDRO, Causal-WDRO, and KL-DRO.
'''

import os
from CausalSDRO.Tools.Parameters import *
from CausalSDRO.Optimizers.Compared_DRO_Optimizer import *

'''
    Changeable Parameters
'''
# Select decision rule(s)
USE_SRF = True

# SRF hyperparameters, d epth and tree number
SRF_depth = 5
Tree_number = 20
# Epochs for SAA and SCSC
EPOCHS_SAA  = 300
EPOCHS_SCSC = 300
# Parameter for TNN. We don't use it here.
hidden_layer_dim = 10

## Training sample size
n_train_list = [100, 200, 400, 800]
## Feature dimension
d_x_list = [3, 5, 10, 20]
# Demand and Decision dimensions
d_z = 3
d_y = 3
# Number of instances for each scale
n_instances = 10
# Testing sample size for each instance
n_test = 10000

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
Use_Diverse_SRT = False
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
OUTPUT_DIR = "../Outputs/Outputs_DRO_Comparison"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

file_name = "DRO_Compare_Inventory_results.xlsx"

'''
    Main Function
'''
if __name__ == "__main__":
    # Record all results and output as an excel
    all_results = []
    # Scale
    for n_train in n_train_list:
        for d_x in d_x_list:

            # Set SRF depth
            SRF_depth = math.ceil(math.log2(d_x)) + 1
            # Update parameters
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

                F_star = 3.29

                inv_params = {
                    'problem_name': 'Inventory',
                    'h_cost': h_cost_tensor,
                    'b_cost': b_cost_tensor,
                    'c_cost': c_cost_tensor,
                    's_cost': s_cost_tensor,
                    'lambda': 0, # To be modified in Compared_DRO_Optimizer
                    'epsilon': 0, # To be modified in Compared_DRO_Optimizer
                    'norm': 0, # To be modified in Compared_DRO_Optimizer
                    'dimension': [d_x, d_y, d_z]
                }

                # For each instance:
                for instance in range(1, 1 + n_instances):
                    print(f"--- Instance {instance}/{n_instances} ---")

                    # Solve all DRO models
                    optimizer = Compared_DRO_Optimizer(F_star, n_train, n_test, instance, params, inv_params, device)

                    result = optimizer.solve_all_models()

                    all_results.append(result)

    # --- Save all results to Excel ---
    df_results = pd.DataFrame(all_results)

    if not df_results.empty:
        print("\nSaving results to Excel...")

        columns_ordered = ['Model', 'Parameters', 'N', 'D_X', 'Instance', 'F*', 'F_ERM',
                           'F_1_CSDRO', 'F_2_CSDRO', 'F_KLDRO', 'F_1_SDRO', 'F_2_SDRO', 'F_1_CWDRO', 'F_2_CWDRO',
                           'P_1-Causal-SDRO', 'P_2-Causal-SDRO', 'P_1-SDRO', 'P_2-SDRO', 'P_1-Causal-WDRO', 'P_2-Causal-WDRO', 'P_KL-DRO']

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