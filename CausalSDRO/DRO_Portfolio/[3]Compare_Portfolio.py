'''
    This file compares Causal-SDRO, SDRO, Causal-WDRO, KL-DRO, and Smooth.
'''
import os
import sys
import math
from pathlib import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd
import torch
from Tools.Parameters import *
from Optimizers.Portfolio_Optimizer import *

'''
    Choose trade dates for decision
'''
set_trade_dates = ['2021-01-03', '2021-04-01', '2021-07-01', '2021-10-01',
                   '2022-01-03', '2022-04-01', '2022-07-01', '2022-10-03']

'''
    Changeable Parameters
'''
# SRF hyperparameters
Tree_number = 20
hidden_layer_dim = 10

# Training samples and dimensions
HISTORY_DAYS = 730
TRAIN_DAYS = 730
RANDOM_SEED = 42
STOCK_NUM = 50
d_x = 5
d_y = STOCK_NUM
d_z = STOCK_NUM + 1
SRF_depth = math.ceil(math.log2(d_x)) + 1

# Portfolio and DRO parameters
ETA_List = [0.25, 0.5, 0.75, 1]
p_norm_list = [1, 2]
n_m_list = [1, 10]
statistics_n_m_list = [1, 10]
BASELINE_LR = 0.01
LOSS_BOUND = 500.0
SOFTPLUS_A = 2.0

'''
    Basic Settings
'''
Use_Linear_Leaf = False
Use_Diverse_SRT = False

params = Parameters_Portfolio()
params.B = LOSS_BOUND
params.a = SOFTPLUS_A
params.setModelStrcuture(hidden_layer_dim, SRF_depth, Tree_number,
                        Use_Linear_Leaf, Use_Diverse_SRT)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PARAMETER_FILE = PROJECT_DIR / 'Outputs/Outputs_Tuning/DRO_Portfolio_Best_Parameters.xlsx'
OUTPUT_DIR = PROJECT_DIR / 'Outputs/Outputs_DRO_Comparison'
STATISTICS_DIR = PROJECT_DIR / 'Outputs/Outputs_CSDRO/Portfolio_3_2_WDROWarmStart'
file_name = 'DRO_Compare_Portfolio_results.xlsx'

'''
    Read Data
'''
DATA_FILE = PROJECT_DIR / 'DRO_Portfolio/data/[Return and Features]20170101-20230331.csv'
all_data = pd.read_csv(DATA_FILE, index_col=0)

# Fixed model parameters and learning rates.
best_parameters = pd.read_excel(PARAMETER_FILE, sheet_name='Comparison_Parameters')
if not best_parameters['N'].eq(TRAIN_DAYS).all() or RANDOM_SEED != 42:
    raise ValueError('The parameter table was selected for N=730 and seed=42.')

'''
    Main Function
'''
if __name__ == '__main__':
    torch.set_num_threads(1)
    print(f'Using device: {device}')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATISTICS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_statistics = []
    for n_m in n_m_list:
        for date in set_trade_dates:
            for ETA in ETA_List:
                print(f'n_m={n_m}, date={date}, omega={ETA}', flush=True)
                pf_params = {
                    'problem_name': 'Portfolio',
                    'eta': ETA,
                    'date': date,
                    'historical_days': HISTORY_DAYS,
                    'train_days': TRAIN_DAYS,
                    'seed': RANDOM_SEED,
                    'n_m': n_m,
                    'dimension': [d_x, d_y, d_z],
                    'p_norm_list': p_norm_list,
                    'baseline_lr': BASELINE_LR,
                    'data': all_data
                }
                optimizer = Portfolio_Optimizer(params, pf_params, best_parameters, device)
                result, statistics = optimizer.solve_all_models()
                all_results.append(result)
                all_statistics.append(statistics)

    # Save the comparison table.
    df_results = pd.DataFrame(all_results)
    if not df_results.empty:
        df_results = df_results.sort_values(['n_m', 'Parameters', 'Instance'])
        excel_path = OUTPUT_DIR / file_name
        df_results.to_excel(excel_path, index=False, sheet_name='All_Results')
        print(f'Excel -> {excel_path}')

    # Save financial statistics for each omega and grouping size.
    for n_m in statistics_n_m_list:
        for ETA in ETA_List:
            statistics = [row for row in all_statistics if row['n_m'] == n_m and row['eta'] == ETA]
            if statistics:
                df_statistics = pd.DataFrame(statistics).sort_values('date').set_index('date', drop=False).T
                excel_path = STATISTICS_DIR / f'portfolio_all_results_eta={ETA}_nm={n_m}.xlsx'
                df_statistics.to_excel(excel_path, index=True)

    print('All tasks completed.')
