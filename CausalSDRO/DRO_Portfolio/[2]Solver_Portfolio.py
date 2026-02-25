'''
   This file solves the Feature-based Portfolio Problem
'''

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time

from CausalSDRO.Optimizers.Portfolio_Optimizer import *

from CausalSDRO.Tools.Functions import *
from CausalSDRO.Tools.Plotting import *
from CausalSDRO.Tools.Parameters import *

'''
    Choose trade dates for decision
'''

set_trade_dates = ['2021-01-03', '2021-04-01', '2021-07-01', '2021-10-01',
                   '2022-01-03', '2022-04-01', '2022-07-01', '2022-10-03']

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
EPOCHS_SAA  = 400
EPOCHS_SCSC = 400

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
OUTPUT_DIR = "../Outputs/Outputs_CSDRO"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

file_name = "DRO_Portfolio_all_results.xlsx"

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
    K = EPOCHS_SCSC * (HISTORY_DAYS // params.BATCH_SIZE)
    # For SCSC, alpha and beta is proportional to O(1/k^{-1/2})
    LR_alpha = 1 / np.sqrt(K + 1e-9)
    LR_beta =  2 / np.sqrt(K + 1e-9)
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

            print('##### Date = ', date, ' #####')

            # Set decision rule parameters
            if USE_TwoLayerNN == True:
                # Number of neurons is much lager than d_x
                hidden_layer_dim = d_x * 50

            if USE_SRF == True:
                SRF_depth = math.ceil(math.log2(d_x)) + 1

            params.setModelStrcuture(hidden_layer_dim, SRF_depth, Tree_number, Use_Linear_Leaf, Use_Diverse_SRT)

            # set portfolio parameters
            pf_params = {'problem_name': 'Portfolio',
                        'eta': ETA,
                        'historical_days': HISTORY_DAYS,
                        'lambda': 0,
                        'epsilon': 0,
                        'norm': 0,
                        'dimension': [d_x, d_y, d_z]
                        }

            start_time = time.time()

            results = solve_all_models(date, x_features, y_ret_select, pf_params, params, device)
            all_results.extend(results)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Run Time: {elapsed_time:.2f} seconds.")

        output_path = os.path.join(OUTPUT_DIR, f"portfolio_all_results_eta={ETA}.xlsx")

        results_df = pd.DataFrame(all_results).T
        results_df.to_excel(output_path, index=True)

        print(f"Excel -> {output_path}")