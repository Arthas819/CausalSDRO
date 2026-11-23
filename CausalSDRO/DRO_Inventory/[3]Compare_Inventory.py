'''
    This file compares Causal-SDRO, SDRO, Causal-WDRO, KL-DRO, and Smooth.
'''

import copy
import math
import os
import sys
import runpy
from pathlib import *

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from Tools.Parameters import *
from Optimizers.Compared_DRO_Optimizer import *
import pandas as pd
import torch

'''
    Changeable Parameters
'''
# Training and test samples
n_train_list = [20, 40, 80, 100]
d_x_list = [3, 5, 10]
n_m_list = [1, 5, 10]
n_instances = 10
n_test = 10000

# SRF and training parameters
Tree_number = 20
EPOCHS_SAA = 300
EPOCHS_SCSC = 300

# Inventory parameters
d_y = d_z = 3
h_cost = [1, 0.7, 0.6]
b_cost = [1.8, 1.6, 1.2]
c_cost = [0, 0, 0]
s_cost = [[0, 1.7, 2],
          [float('inf'), 0, 1.5],
          [float('inf'), float('inf'), 0]]

# Compare completed CausalSDRO fits on validation data.
WARM_VALIDATION_SIZE = 1000
WARM_LR_FACTORS = (0.03, 0.1, 0.3)

'''
    Basic Settings
'''
Use_Linear_Leaf = False
Use_Diverse_SRT = True
params = Parameters_Inv()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUTPUT_DIR = os.path.join(PROJECT_DIR, 'Outputs/Outputs_DRO_Comparison')
file_name = 'DRO_Compare_Inventory_results.xlsx'

# Output column order; omit raw values and validation diagnostics.
columns_ordered = ['Model', 'Parameters', 'N', 'D_X', 'n_m', 'M', 'Instance', 'F*', 'F_ERM']
excluded_columns = [
    'Validation_1_Smooth', 'Validation_1_CSDRO',
    'Smooth_1_Selected_Step', 'CSDRO_1_Selected_Step', 'CSDRO_1_LR_Factor',
    'Validation_2_Smooth', 'Validation_2_CSDRO',
    'Smooth_2_Selected_Step', 'CSDRO_2_Selected_Step', 'CSDRO_2_LR_Factor'
]

'''
    Main Function
'''
if __name__ == '__main__':
    all_results = []
    for n_m in n_m_list:
        for n_train in n_train_list:
            for d_x in d_x_list:
                for instance in range(1, n_instances + 1):
                    print(f'N={n_train}, d_x={d_x}, n_m={n_m}, instance={instance}', flush=True)
                    if d_x == 20:
                        raise ValueError('d_x=20 is excluded from this comparison.')

                    # Each instance uses its own parameters.
                    fixed_params = copy.deepcopy(params)
                    fixed_params.setEpochs(EPOCHS_SAA, EPOCHS_SCSC)
                    fixed_params.WARM_VALIDATION_SIZE = WARM_VALIDATION_SIZE
                    fixed_params.WARM_LR_FACTORS = WARM_LR_FACTORS

                    SRF_depth = math.ceil(math.log2(d_x)) + 1
                    hidden_layer_dim = 64 * d_x
                    fixed_params.setModelStrcuture(hidden_layer_dim, SRF_depth, Tree_number,
                                                  Use_Linear_Leaf, Use_Diverse_SRT)

                    inv_params = {
                        'problem_name': 'Inventory',
                        'dimension': [d_x, d_y, d_z],
                        'n_m': n_m,
                        'seed': 42 + instance - 1,
                        'mu': fixed_params.mu,
                        'B': fixed_params.B,
                        'a': fixed_params.a,
                        'x_bound': fixed_params.x_bound,
                        'z_bound': fixed_params.z_bound,
                        'lambda': 0,
                        'epsilon': 0,
                        'norm': 0,
                        'h_cost': torch.tensor(h_cost, device=device, dtype=torch.float64),
                        'b_cost': torch.tensor(b_cost, device=device, dtype=torch.float64),
                        'c_cost': torch.tensor(c_cost, device=device, dtype=torch.float64),
                        's_cost': torch.tensor(s_cost, device=device, dtype=torch.float64)
                    }
                    optimizer = Compared_DRO_Optimizer(None, n_train, n_test, instance,
                                                       fixed_params, inv_params, device)

                    result = optimizer.solve_all_models()
                    all_results.append(result)

                    # Save after each instance, so completed results remain available.
                    df_results = pd.DataFrame(all_results)
                    final_columns = [col for col in df_results.columns
                                     if 'raw' not in col.lower() and col not in excluded_columns]
                    df_results = df_results[final_columns]
                    final_columns = [col for col in columns_ordered if col in df_results.columns]
                    final_columns += [col for col in df_results.columns if col not in final_columns]
                    df_to_save = df_results[final_columns]

                    excel_path = Path(OUTPUT_DIR) / file_name
                    excel_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary_path = excel_path.with_name(excel_path.stem + '.tmp.xlsx')
                    df_to_save.to_excel(temporary_path, index=False, sheet_name='All_Results')
                    temporary_path.replace(excel_path)
    plotting = runpy.run_path(os.path.join(PROJECT_DIR, 'DRO_Inventory/[4]Visualization_Inventory.py'))
    plotting['plot_comparison'](pd.DataFrame(all_results))
    df_results = pd.DataFrame(all_results)
