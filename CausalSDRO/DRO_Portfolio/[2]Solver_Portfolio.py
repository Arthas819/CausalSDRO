'''
   This file solves the Feature-based Portfolio Problem
'''

import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time

from Optimizers.Portfolio_Optimizer import *

from Tools.Functions import *
from Tools.Plotting import *
from Tools.Parameters import *

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
# ERM epochs and total TNN Causal-SDRO updates
EPOCHS_SAA = 400
TNN_ITERATIONS = 1200

# Historical observations before each trade date
HISTORY_DAYS = 730
TRAIN_DAYS = 730
RANDOM_SEED = 42
# Dimensions
STOCK_NUM = 50
d_x = 5
d_y = STOCK_NUM
d_z = STOCK_NUM + 1

# Portfolio and DRO parameters
ETA_List = [5]
p_norm_list = [1, 2]
n_m_list = [1, 5, 10]

'''
    Basic Settings
'''
# Choose SRF type
Use_Linear_Leaf = False # False maybe better
# Use feature subsets in each SRT?
Use_Diverse_SRT = False

# Fixed parameters
params = Parameters_Portfolio()
params.setEpochs(EPOCHS_SAA, TNN_ITERATIONS)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory
OUTPUT_DIR = os.path.join(PROJECT_DIR, "Outputs/Outputs_CSDRO")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

file_name = "DRO_Portfolio_all_results.xlsx"

'''
    Read Data
'''

all_data = pd.read_csv(os.path.join(PROJECT_DIR, 'DRO_Portfolio/data/[Return and Features]20170101-20230331.csv'))
all_data = all_data.set_index(all_data.columns[0])
y_returns = all_data.iloc[:, :-5]
x_features = all_data.iloc[:, -5:]


'''
    Main Function
'''
if __name__ == "__main__":

    # Set initial learning rates for SCSC
    K = TNN_ITERATIONS
    # For SCSC, alpha and beta is proportional to O(1/k^{-1/2})
    LR_alpha = 1 / np.sqrt(K + 1e-9)
    LR_beta =  2 / np.sqrt(K + 1e-9)
    params.setLearningRate(LR_alpha, LR_beta)

    # Record all results and output as an excel
    all_results = []
    comparison_results = []

    # Select Stocks Randomly
    rng = np.random.RandomState(RANDOM_SEED)
    selected_cols = rng.choice(y_returns.columns, size=STOCK_NUM, replace=False)
    y_ret_select = y_returns[selected_cols]

    for n_m in n_m_list:

        # Scale
        for ETA in ETA_List:
            all_results = []

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
                            'historical_days': HISTORY_DAYS, 'train_days': TRAIN_DAYS,
                            'n_m': n_m, 'B': params.B, 'a': params.a, 'z0_bound': params.z0_bound,
                            'lambda': 0,
                            'epsilon': 0,
                            'norm': 0,
                            'dimension': [d_x, d_y, d_z]
                            }

                start_time = time.time()

                results = solve_all_models(date, x_features, y_ret_select, pf_params, params, device,
                                           tnn_iterations=TNN_ITERATIONS)
                all_results.extend(results)

                # Both decision rules share the better ERM loss for this date and n_m.
                for result in results:
                    F_erm_common = min(result['loss_cmv_tnn_saa'], result['loss_cmv_srf_saa'])
                    gap = F_erm_common - result['loss_ph']
                    for norm in p_norm_list:
                        F_tnn = result[f'loss_{norm}_sdmv_tnn_scsc']
                        F_srf = result[f'loss_{norm}_sdmv_srf_scsc']
                        P_tnn = 100 * (F_erm_common - F_tnn) / gap if gap > 1e-10 else np.nan
                        P_srf = 100 * (F_erm_common - F_srf) / gap if gap > 1e-10 else np.nan
                        comparison_results.append({
                            'date': result['date'], 'eta': result['eta'], 'norm': norm,
                            'n_m': result['n_m'], 'N': result['N'], 'M': result['M'],
                            'F*': result['loss_ph'], 'F_ERM': F_erm_common,
                            'F_DRO_2NN': F_tnn, 'F_DRO_SRF': F_srf,
                            'P_2NN': max(-100, P_tnn) if np.isfinite(P_tnn) else np.nan,
                            'P_SRF': max(-100, P_srf) if np.isfinite(P_srf) else np.nan
                        })
                        print(f'{norm}: F_ERM={F_erm_common:.6f}, P_2NN={P_tnn:.2f}%, P_SRF={P_srf:.2f}%')

                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Run Time: {elapsed_time:.2f} seconds.")

            output_path = os.path.join(OUTPUT_DIR, f"portfolio_all_results_eta={ETA}_nm={n_m}.xlsx")

            results_df = pd.DataFrame(all_results).T
            results_df.to_excel(output_path, index=True)

            print(f"Excel -> {output_path}")

    # Summary consumed by the SRF/2NN boxplot, including every date and n_m.
    comparison_path = os.path.join(OUTPUT_DIR, file_name)
    pd.DataFrame(comparison_results).to_excel(comparison_path, index=False)
    print(f"Excel -> {comparison_path}")