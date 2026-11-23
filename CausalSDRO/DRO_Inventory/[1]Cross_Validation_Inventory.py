"""
    Cross validation for inventory substitution.
"""

import os
import sys
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from openpyxl import Workbook

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from Optimizers.COPT_Optimizer import *

from Tools.Data_Generator_Inventory import *
from Tools.Data_Generator_Newsvendor import *
from Tools.Decision_Rules import *
from Tools.Parameters import *
from Optimizers.ERM_Trainer import *
from Optimizers.CSDRO_Trainer import *
from Optimizers.SDRO_Trainer import *
from Optimizers.CWDRO_Trainer import *
from Optimizers.KLDRO_Trainer import *

# Tune lambda and epsilon only.
cv_csdro = True
cv_cwdro = True
cv_kldro = True
lambda_list = [50.0, 100.0, 150.0, 200.0, 250.0]
epsilon_list = [0.2, 0.4, 0.6, 0.8, 1.0]
cwdro_lambda_list = [0.01, 0.1, 0.5, 1.0, 10.0]
kldro_lambda_list = [2.0, 5.0, 10.0, 20.0, 50.0]
p_norm_list = [1, 2]

# One CV scale; reuse its parameters at all experiment sizes.
n_train_list = [40]
d_x_list = [10]
n_m = 1
d_y = d_z = 3
n_instances = 2
n_test = 10000
EPOCHS_SAA = 300
EPOCHS_SCSC = 1200
EPOCHS_KLDRO = 300
Tree_number = 20
Use_Linear_Leaf = False
Use_Diverse_SRT = True

h_cost = [1, 0.7, 0.6]
b_cost = [1.8, 1.6, 1.2]
c_cost = [0, 0, 0]
s_cost = [[0, 1.7, 2], [float('inf'), 0, 1.5], [float('inf'), float('inf'), 0]]

params = Parameters_Inv()
params.setEpochs(EPOCHS_SAA, EPOCHS_SCSC)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'Outputs/Outputs_Tuning')
output_filename = 'DRO_Inventory_Best_Parameters.xlsx'


def set_csdro_steps(n_batches):
    # Fixed training settings; previous CV outputs must not override them.
    params.EPOCHS_SCSC = EPOCHS_SCSC
    params.setCSDROParameters(n_batches)


def save_cv_results(results):
    details = pd.DataFrame(results)
    keys = ['Model', 'norm', 'lambda', 'epsilon']
    summary = details.groupby(keys, as_index=False)[['F_star', 'F_erm', 'F_model', 'P_raw', 'P']].mean()
    best_parameters = []
    for (model_name, norm), subset in summary.groupby(['Model', 'norm']):
        valid = subset.dropna(subset=['P_raw'])
        if valid.empty:
            continue
        best = valid.loc[valid['P_raw'].idxmax()]
        row = {'Model': model_name, 'norm': norm, 'best_lambda': best['lambda'],
               'best_epsilon': best['epsilon'], 'best_P': best['P_raw']}
        if model_name in ('Causal-SDRO', 'SDRO'):
            matching = details[(details['Model'] == model_name) & (details['norm'] == norm) &
                               (details['lambda'] == best['lambda']) & (details['epsilon'] == best['epsilon'])].iloc[0]
            row.update({'K': matching['K'], 'Q': matching['Q'],
                        'best_alpha_scale': matching['alpha_scale'], 'best_beta_scale': matching['beta_scale']})
        best_parameters.append(row)
        pivot = subset.pivot(index='lambda', columns='epsilon', values='P_raw').sort_index()
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                         cbar_kws={'label': 'Out-of-sample Performance (%)'})
        ax.invert_yaxis()
        plt.xlabel('epsilon' if model_name in ('Causal-SDRO', 'SDRO') else 'epsilon (not applicable)')
        plt.ylabel('lambda')
        ax.set_xticklabels([f'{value:.1f}' for value in pivot.columns], rotation=0)
        ax.set_yticklabels([f'{value:g}' for value in pivot.index], rotation=0)
        plt.tight_layout()
        filename = os.path.join(OUTPUT_DIR, f'Inventory-{model_name}-{norm}_Heatmap')
        plt.savefig(filename + '.png')
        plt.savefig(filename + '.pdf')
        plt.close()
    best = pd.DataFrame(best_parameters)
    path = os.path.join(OUTPUT_DIR, output_filename)
    if os.path.exists(path):
        previous = pd.read_excel(path)
        previous = previous[~previous['Model'].isin(details['Model'].unique())]
        if cv_csdro:
            previous = previous[previous['Model'] != 'SDRO']
        best = pd.concat([previous.reindex(columns=best.columns), best], ignore_index=True)
    workbook = Workbook()
    workbook.remove(workbook.active)
    for name, frame in [('Sheet1', best), ('CV_Summary', summary), ('CV_Details', details)]:
        sheet = workbook.create_sheet(name)
        sheet.append(list(frame.columns))
        for values in frame.itertuples(index=False, name=None):
            sheet.append([None if pd.isna(value) else value for value in values])
        for column in sheet.columns:
            sheet.column_dimensions[column[0].column_letter].width = max(
                len(str(cell.value)) if cell.value is not None else 0 for cell in column) + 2
    workbook.save(path)
    print(best.to_string(index=False))
    return best


if __name__ == '__main__':
    get_copt_env()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'CV: K={EPOCHS_SCSC}, alpha_scale={params.alpha_scale:g}, beta_scale={params.beta_scale:g}')
    results = []
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
            data_cache = {}
            for instance in range(1, n_instances + 1):
                seed = 42 + instance - 1
                inv_params['seed'] = seed
                set_seed(seed)
                data_gen = DataGenerator_Inventory(n_train, n_test, inv_params, device)
                X_train, Y_train, X_test, Y_test, loader = data_gen.get_data(params.BATCH_SIZE)
                F_star = calculate_F_star_theoretical_is(data_gen, X_test, Y_test, inv_params)
                set_seed(seed)
                initial = SRF(d_x, depth, d_z, Tree_number, params).to(device).double()
                set_seed(seed)
                trainer = ERM_Trainer(copy.deepcopy(initial), loader, params, device, 'SRF')
                model_erm, _ = trainer.train_erm(X_train, Y_train, inv_params)
                F_erm = get_test_loss(model_erm, X_test, Y_test, inv_params)
                data_cache[seed] = (X_train, Y_train, X_test, Y_test, loader, initial, F_star, F_erm)
            combinations = []
            for norm in p_norm_list:
                if cv_csdro:
                    combinations += [('Causal-SDRO', norm, lam, eps)
                                     for lam in lambda_list for eps in epsilon_list]
                if cv_cwdro:
                    combinations += [('Causal-WDRO', norm, lam, 0.0) for lam in cwdro_lambda_list]
            if cv_kldro:
                combinations += [('KL-DRO', 0, lam, 0.0) for lam in kldro_lambda_list]
            for model_name, norm, lam, eps in combinations:
                for seed, data in data_cache.items():
                    X_train, Y_train, X_test, Y_test, loader, initial, F_star, F_erm = data
                    inv_params.update({'seed': seed, 'norm': norm, 'lambda': lam, 'epsilon': eps})
                    params.EPOCHS_SCSC = EPOCHS_SCSC
                    set_seed(seed)
                    if model_name == 'Causal-SDRO':
                        set_csdro_steps(len(loader))
                        trainer = CSDRO_Trainer(copy.deepcopy(initial), loader, params, device, 'SRF')
                        model, _ = trainer.train_scsc(X_train, Y_train, inv_params)
                    elif model_name == 'Causal-WDRO':
                        trainer = CWDRO_Trainer(copy.deepcopy(initial), loader, params, device, 'SRF')
                        model, _ = trainer.train_rtmlmc(X_train, Y_train, inv_params)
                    else:
                        params.EPOCHS_SCSC = EPOCHS_KLDRO
                        trainer = KLDRO_Trainer(copy.deepcopy(initial), loader, params, device, 'SRF')
                        model, _ = trainer.train_sgd(X_train, Y_train, inv_params)
                    F_model = get_test_loss(model, X_test, Y_test, inv_params)
                    gap = F_erm - F_star
                    P_raw = 100 * (F_erm - F_model) / gap if gap > 1e-10 else np.nan
                    K = trainer.K if model_name == 'Causal-SDRO' else trainer.epochs * (len(loader) if model_name == 'KL-DRO' else 1)
                    results.append({'Model': model_name, 'norm': norm, 'lambda': lam, 'epsilon': eps,
                                    'K': K, 'Q': trainer.Q if model_name == 'Causal-SDRO' else np.nan,
                                    'alpha_scale': trainer.alpha_k * K ** 0.5 if model_name == 'Causal-SDRO' else np.nan,
                                    'beta_scale': trainer.beta_k * K ** 0.5 if model_name == 'Causal-SDRO' else np.nan,
                                    'seed': seed, 'N': n_train, 'd_x': d_x, 'n_m': n_m,
                                    'F_star': F_star, 'F_erm': F_erm, 'F_model': F_model,
                                    'P_raw': P_raw, 'P': max(-100, P_raw) if np.isfinite(P_raw) else np.nan})
                    print(f'{model_name} p={norm}, lambda={lam}, epsilon={eps}, seed={seed}, P={P_raw:.2f}%')
            if cv_csdro:
                # SDRO reuses Causal-SDRO parameters; evaluate its own performance.
                summary = pd.DataFrame(results)
                for norm in p_norm_list:
                    subset = summary[(summary['Model'] == 'Causal-SDRO') & (summary['norm'] == norm)]
                    scores = subset.groupby(['lambda', 'epsilon'])['P_raw'].mean()
                    lam, eps = scores.idxmax()
                    for seed, data in data_cache.items():
                        X_train, Y_train, X_test, Y_test, loader, initial, F_star, F_erm = data
                        inv_params.update({'seed': seed, 'norm': norm, 'lambda': lam, 'epsilon': eps})
                        params.EPOCHS_SCSC = EPOCHS_SCSC
                        set_csdro_steps(len(loader))
                        set_seed(seed)
                        trainer = SDRO_Trainer(copy.deepcopy(initial), loader, params, device, 'SRF')
                        model, _ = trainer.train_scsc(X_train, Y_train, inv_params)
                        F_model = get_test_loss(model, X_test, Y_test, inv_params)
                        gap = F_erm - F_star
                        P_raw = 100 * (F_erm - F_model) / gap if gap > 1e-10 else np.nan
                        results.append({'Model': 'SDRO', 'norm': norm, 'lambda': lam, 'epsilon': eps,
                                        'K': trainer.K, 'Q': trainer.Q,
                                        'alpha_scale': trainer.alpha_k * trainer.K ** 0.5,
                                        'beta_scale': trainer.beta_k * trainer.K ** 0.5, 'seed': seed, 'N': n_train,
                                        'd_x': d_x, 'n_m': n_m, 'F_star': F_star, 'F_erm': F_erm,
                                        'F_model': F_model, 'P_raw': P_raw,
                                        'P': max(-100, P_raw) if np.isfinite(P_raw) else np.nan})
                        print(f'SDRO p={norm}, seed={seed}, P={P_raw:.2f}%')
    save_cv_results(results)
