"""
    Solve the inventory problem with SRF and 2NN decision rules.
"""

import os
import sys
import copy
import runpy
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from Optimizers.COPT_Optimizer import *

from Optimizers.ERM_Trainer import *
from Optimizers.CSDRO_Trainer import *
from Tools.Data_Generator_Inventory import *
from Tools.Data_Generator_Newsvendor import *
from Tools.Decision_Rules import *
from Tools.Plotting import *
from Tools.Parameters import *

# Decision rules
USE_TwoLayerNN = True
USE_SRF = True
Use_Linear_Leaf = False
Use_Diverse_SRT = True
Tree_number = 20
Convergence_figure = False

# Experiment sizes
n_train_list = [20, 40, 80, 100]
d_x_list = [3, 5, 10, 20]
n_m_list = [1, 5, 10]
p_norm_list = [1, 2]
d_y = d_z = 3
n_instances = 10
n_test = 10000
EPOCHS_SAA = 300
TNN_ITERATIONS = 1200

# Inventory costs
h_cost = [1, 0.7, 0.6]
b_cost = [1.8, 1.6, 1.2]
c_cost = [0, 0, 0]
s_cost = [[0, 1.7, 2], [float('inf'), 0, 1.5], [float('inf'), float('inf'), 0]]

params = Parameters_Inv()
params.setEpochs(EPOCHS_SAA, TNN_ITERATIONS)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'Outputs/Outputs_CSDRO')
file_name = 'DRO_Inventory_all_results.xlsx'
parameter_file = os.path.join(PROJECT_DIR, 'Outputs/Outputs_Tuning/DRO_Inventory_Best_Parameters.xlsx')


if __name__ == '__main__':
    get_copt_env()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_parameters = pd.read_excel(parameter_file)
    all_results = []
    for n_m in n_m_list:
        for n_train in n_train_list:
            for d_x in d_x_list:
                depth = math.ceil(math.log2(d_x)) + 1
                params.setModelStrcuture(64 * d_x, depth, Tree_number,
                                         Use_Linear_Leaf, Use_Diverse_SRT)
                inv_params = {'problem_name': 'Inventory', 'dimension': [d_x, d_y, d_z],
                              'n_m': n_m, 'mu': params.mu, 'B': params.B, 'a': params.a,
                              'x_bound': params.x_bound, 'z_bound': params.z_bound}
                for key, values in [('h_cost', h_cost), ('b_cost', b_cost),
                                    ('c_cost', c_cost), ('s_cost', s_cost)]:
                    inv_params[key] = torch.tensor(values, device=device, dtype=torch.float64)
                for instance in range(1, n_instances + 1):
                    seed = 42 + instance - 1
                    inv_params['seed'] = seed
                    set_seed(seed)
                    data_gen = DataGenerator_Inventory(n_train, n_test, inv_params, device)
                    X_train, Y_train, X_test, Y_test, loader = data_gen.get_data(params.BATCH_SIZE)
                    F_star = calculate_F_star_theoretical_is(data_gen, X_test, Y_test, inv_params)
                    print(f'N={n_train}, d_x={d_x}, n_m={n_m}, instance={instance}, SAA F*={F_star:.6f}')
                    models = []
                    if USE_TwoLayerNN: models.append('TNN')
                    if USE_SRF: models.append('SRF')
                    instance_results = []
                    for model_type in models:
                        set_seed(seed)
                        if model_type == 'TNN':
                            initial = TNN(d_x, 64 * d_x, d_z).to(device).double()
                            model_name = 'TNN'
                        else:
                            initial = SRF(d_x, depth, d_z, Tree_number, params).to(device).double()
                            model_name = 'SRF(Linear Leaf)' if Use_Linear_Leaf else 'SRF(Constant Leaf)'
                        set_seed(seed)
                        trainer = ERM_Trainer(copy.deepcopy(initial), loader, params, device, model_type)
                        model_erm, history_erm = trainer.train_erm(X_train, Y_train, inv_params)
                        F_erm = get_test_loss(model_erm, X_test, Y_test, inv_params)
                        for norm in p_norm_list:
                            # Match method and norm only.
                            row = best_parameters[(best_parameters['Model'] == 'Causal-SDRO') &
                                                  (best_parameters['norm'] == norm)].iloc[0]
                            inv_params.update({'norm': norm, 'lambda': row['best_lambda'],
                                               'epsilon': row['best_epsilon']})
                            params.EPOCHS_SCSC = math.ceil(TNN_ITERATIONS / len(loader))
                            if model_type == 'SRF':
                                params.setCSDROParameters(len(loader), row)
                                params.grad_clip = None
                            else:
                                # Keep the original 2NN step-size scales with the new update budget.
                                K = params.EPOCHS_SCSC * len(loader)
                                params.setLearningRate(1 / K ** 0.5, 2 / K ** 0.5)
                                params.Q = math.ceil(K ** 0.25)
                                params.grad_clip = 4
                            set_seed(seed)
                            trainer = CSDRO_Trainer(copy.deepcopy(initial), loader, params, device, model_type)
                            model_dro, history_dro = trainer.train_scsc(X_train, Y_train, inv_params)
                            F_dro = get_test_loss(model_dro, X_test, Y_test, inv_params)
                            instance_results.append({'Norm': norm, 'lambda': row['best_lambda'],
                                                'epsilon': row['best_epsilon'], 'N': n_train,
                                                'D_X': d_x, 'n_m': n_m, 'M': n_train // n_m,
                                                'Instance': instance, 'Model': model_name,
                                                'F*': F_star, 'F_ERM': F_erm, 'F_DRO': F_dro,
                                                'Regret': F_dro - F_star})
                            if instance == 1 and Convergence_figure:
                                plot_convergence_curve({'ERM': history_erm, 'CM-SCSC': history_dro},
                                    output_dir=OUTPUT_DIR,
                                    img_name=f'conv_N{n_train}_D{d_x}_nm{n_m}_p{norm}_{model_type}')
                    # Both decision rules share the better ERM loss for this instance.
                    if instance_results:
                        F_erm_common = min(result['F_ERM'] for result in instance_results)
                        for result in instance_results:
                            result['F_ERM'] = F_erm_common
                            gap = F_erm_common - result['F*']
                            P_raw = 100 * (F_erm_common - result['F_DRO']) / gap if gap > 1e-10 else np.nan
                            result['P_raw'] = P_raw
                            result['Prescriptiveness'] = max(-100, P_raw) if np.isfinite(P_raw) else np.nan
                            print(f"{result['Norm']}-{result['Model']}: F_DRO={result['F_DRO']:.6f}, "
                                  f"F_ERM={F_erm_common:.6f}, P={P_raw:.2f}%")
                        all_results.extend(instance_results)
                    pd.DataFrame(all_results).to_excel(os.path.join(OUTPUT_DIR, file_name),
                                                       index=False, sheet_name='All_Results')
    plotting = runpy.run_path(os.path.join(PROJECT_DIR, 'DRO_Inventory/[4]Visualization_Inventory.py'))
    plotting['plot_results'](pd.DataFrame(all_results))
