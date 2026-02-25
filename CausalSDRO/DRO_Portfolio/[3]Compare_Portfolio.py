'''
   This file compares Causal-SDRO, SDRO, Causal-WDRO, and KL-DRO.
'''

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from CausalSDRO.Tools.Parameters import *
from CausalSDRO.Optimizers.Compared_DRO_Optimizer import *
from CausalSDRO.Optimizers.Portfolio_Optimizer import *

'''
    Choose trade dates for decision
'''

set_trade_dates = ['2021-01-03', '2021-04-01', '2021-07-01', '2021-10-01',
                   '2022-01-03', '2022-04-01', '2022-07-01', '2022-10-03']

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

# Training sample size
HISTORY_DAYS = 730
RANDOM_SEED = 42
# Dimensions
STOCK_NUM = 50
d_x = 5
d_y = STOCK_NUM
d_z = STOCK_NUM + 1

# Portfolio and DRO parameters
ETA_List = [1, 3, 5, 7, 9]
p_norm_list = [1, 2]
best_parameters = pd.read_excel("../Outputs/Outputs_Tuning/DRO_Portfolio_Best_Parameters.xlsx")

'''
    Basic Settings
'''
# Choose SRF type
Use_Linear_Leaf = False # False maybe better
# Use feature subsets in each SRT?
Use_Diverse_SRT = False

# Fixed parameters
params = Parameters_Portfolio()
params.setEpochs(EPOCHS_SAA, EPOCHS_SCSC)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory
OUTPUT_DIR = "../Outputs/Outputs_DRO_Comparison"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

file_name = "DRO_Compare_Portfolio_results.xlsx"

'''
    Read Data
'''

all_data = pd.read_csv('data/[Return and Features]20170101-20230331.csv')
all_data = all_data.set_index(all_data.columns[0])
y_returns = all_data.iloc[:, :-5]
x_features = all_data.iloc[:, -5:]

'''
    Main Function
'''
if __name__ == "__main__":

    # Set initial learning rates for SCSC
    # In each epoch, the parameters are updated (n_train / batch_size) times, and thus the total K is
    K = EPOCHS_SCSC * (HISTORY_DAYS // params.BATCH_SIZE)
    # For SCSC, alpha and beta is proportional to O(1/k^{-1/2})
    LR_alpha = 1 / np.sqrt(K + 1e-9)
    LR_beta = 2 / np.sqrt(K + 1e-9)
    params.setLearningRate(LR_alpha, LR_beta)

    # Record all results and output as an excel
    all_results = []

    # Select Stocks Randomly
    rng = np.random.RandomState(RANDOM_SEED)
    selected_cols = rng.choice(y_returns.columns, size=STOCK_NUM, replace=False)
    y_ret_select = y_returns[selected_cols]

    # Scale
    for ETA in ETA_List:

        print('------ ETA=', ETA, ' -------')

        for date in set_trade_dates:

            # Set SRF depth
            SRF_depth = math.ceil(math.log2(d_x)) + 1
            # Update parameters
            params.setModelStrcuture(hidden_layer_dim, SRF_depth, Tree_number, Use_Linear_Leaf, Use_Diverse_SRT)

            # Future return
            future_return = y_ret_select[date:].head(60)

            # set portfolio parameters
            pf_params = {'problem_name': 'Portfolio',
                        'eta': ETA,
                        'historical_days': HISTORY_DAYS,
                        'lambda': 0,
                        'epsilon': 0,
                        'norm': 0,
                        'dimension': [d_x, d_y, d_z],
                        'future_return': future_return
                        }

            # Baseline, Post-hoc (Complete Information) Method
            weight_post_hc = post_hc_solver(future_return.values, ETA)
            return_post_hc = (future_return * weight_post_hc).sum(axis=1)
            F_star, _, _, _, _ = metrics(return_post_hc, ETA)

            print(f"--- Date {date} ---")

            # Solve all DRO models
            optimizer = Compared_DRO_Optimizer(F_star, HISTORY_DAYS, 1, date, params, pf_params, device)

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