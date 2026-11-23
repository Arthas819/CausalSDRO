"""
    This file provides a cross validation procedure for newsvendor.
"""

import os
import sys
import copy
import time
import pandas as pd
from openpyxl import Workbook
import seaborn as sns

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from Optimizers.ERM_Trainer import *
from Optimizers.CSDRO_Trainer import *
from Optimizers.CWDRO_Trainer import *
from Optimizers.KLDRO_Trainer import *
from Tools.Data_Generator_Newsvendor import *
from Tools.Decision_Rules import *
from Tools.Functions import *
from Tools.Plotting import *
from Tools.Parameters import *

# Select models
cv_csdro = True
cv_cwdro = False
cv_kldro = False

# Cross validation searches only lambda and epsilon.
lambda_list = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0]
epsilon_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 2.0, 5.0]
cwdro_lambda_list = [0.01, 0.03, 0.1, 0.3, 0.6, 1.0, 2.0, 5.0, 10.0]
kldro_lambda_list = [0.1, 0.3, 0.6, 1.0, 2.0, 5.0, 10.0]
# Fixed coefficients selected previously; alpha and beta scale as 1/sqrt(K).
alpha_scale_by_norm = {1: 40.0, 2: 48.0}
beta_scale_fixed = 32.0

p_norm_list = [1, 2]
USE_SRF = True
Use_Linear_Leaf = False
Use_Diverse_SRT = False
hidden_layer_dim = 512
SRF_depth = 5
Tree_number = 20
EPOCHS_SAA = 300
EPOCHS_SCSC = 1200
EPOCHS_KLDRO = 300
n_train_list = [40]
d_x_list = [10]
# Number of demand observations per distinct covariate.
n_m = 5
d_z = 1
d_y = 1
n_instances = 3
n_test = 10000
h_val = 0.6
b_val = 1.0
sigma = 1

set_seed(42)
params = Parameters_News()
params.setEpochs(EPOCHS_SAA, EPOCHS_SCSC)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
output_filename = "DRO_Newsvendor_Best_Parameters.xlsx"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "Outputs/Outputs_Tuning")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_cv_results(results):
    best_parameters = []
    details = pd.DataFrame(results)
    group_columns = ['Model', 'norm', 'N', 'd_x', 'n_m', 'lambda', 
                     'epsilon', 'alpha_scale', 'beta_scale', 'learning_rate', 'K', 'Q']
    summary = details.groupby(group_columns, dropna=False).agg(
        P_raw=('P_raw', 'mean'), F_star=('F_star', 'mean'),
        F_erm=('F_erm', 'mean'), F_model=('F_model', 'mean'), instances=('seed', 'count')).reset_index()
    for (model_name, norm, n_train, d_x, n_m), subset in summary.groupby(['Model', 'norm', 'N', 'd_x', 'n_m']):
        if subset['P_raw'].isna().all():
            raise ValueError(f'No valid ERM-to-oracle gap for {model_name}, norm={norm}.')
        best = subset.loc[subset['P_raw'].idxmax()]
        row = {'Model': model_name, 'norm': norm, 'best_lambda': best['lambda'],
               'best_epsilon': best['epsilon'], 'best_P': best['P_raw'],
               'best_alpha_scale': best['alpha_scale'], 'best_beta_scale': best['beta_scale'],
               'K': best['K'], 'Q': best['Q'], 'N': n_train, 'd_x': d_x, 'n_m': n_m,
               'F_star': best['F_star'], 'F_erm': best['F_erm'], 'F_model': best['F_model']}
        best_parameters.append(row)
        if model_name == 'Causal-SDRO':
            best_parameters.append({'Model': 'SDRO', 'norm': norm, 'best_lambda': best['lambda'],
                                    'best_epsilon': best['epsilon'], 'best_alpha_scale': best['alpha_scale'],
                                    'best_beta_scale': best['beta_scale'], 'K': best['K'], 'Q': best['Q'],
                                    'N': n_train, 'd_x': d_x, 'n_m': n_m})
            best_parameters.append({'Model': 'Smooth', 'norm': norm, 'best_epsilon': best['epsilon'],
                                    'N': n_train, 'd_x': d_x, 'n_m': n_m})
        # Heatmaps show mean out-of-sample performance P (%).
        pivot = subset.pivot(index='lambda', columns='epsilon', values='P_raw').sort_index()
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                         cbar_kws={'label': 'Out-of-sample Performance (%)'})
        ax.invert_yaxis()
        ax.set_xticklabels([f'{value:g}' for value in pivot.columns])
        plt.xlabel('epsilon (0 = not applicable)' if model_name != 'Causal-SDRO' else 'epsilon')
        plt.ylabel('lambda')
        plt.tight_layout()
        filename = f'Newsvendor-{model_name}-{norm}_Heatmap'
        plt.savefig(os.path.join(OUTPUT_DIR, filename + '.png'))
        plt.savefig(os.path.join(OUTPUT_DIR, filename + '.pdf'))
        plt.close()

    df_params = pd.DataFrame(best_parameters)
    file_path = os.path.join(OUTPUT_DIR, output_filename)
    if os.path.exists(file_path):
        previous_sheets = pd.read_excel(file_path, sheet_name=None)
        previous = previous_sheets['Sheet1']
        parameter_keys = ['Model', 'norm']
        current_keys = pd.MultiIndex.from_frame(df_params[parameter_keys])
        previous = previous[~pd.MultiIndex.from_frame(previous[parameter_keys]).isin(current_keys)]
        df_params = pd.concat([previous, df_params], ignore_index=True)
        scope_keys = ['Model', 'norm', 'N', 'd_x', 'n_m']
        current_scopes = pd.MultiIndex.from_frame(details[scope_keys].drop_duplicates())
        for sheet_name, current in [('CV_Summary', summary), ('CV_Details', details)]:
            if sheet_name in previous_sheets:
                previous = previous_sheets[sheet_name]
                previous = previous[~pd.MultiIndex.from_frame(previous[scope_keys]).isin(current_scopes)]
                merged = pd.concat([previous, current], ignore_index=True)
                if sheet_name == 'CV_Summary':
                    summary = merged
                else:
                    details = merged
    details.to_csv(os.path.join(OUTPUT_DIR, 'Newsvendor_All_CV_Details.csv'), index=False)
    summary.to_csv(os.path.join(OUTPUT_DIR, 'Newsvendor_All_CV_Summary.csv'), index=False)
    workbook = Workbook()
    workbook.remove(workbook.active)
    for name, frame in [('Sheet1', df_params), ('CV_Summary', summary), ('CV_Details', details)]:
        sheet = workbook.create_sheet(name)
        sheet.append(list(frame.columns))
        for values in frame.itertuples(index=False, name=None):
            sheet.append([None if pd.isna(value) else value for value in values])
    workbook.save(file_path)
    print(df_params.to_string(index=False), flush=True)
    return df_params


if __name__ == "__main__":
    h_cost_tensor = torch.tensor([h_val], device=device, dtype=torch.float64)
    b_cost_tensor = torch.tensor([b_val], device=device, dtype=torch.float64)
    results = []
    best_parameters = []

    for n_train in n_train_list:
        for d_x in d_x_list:
            SRF_depth = math.ceil(math.log2(d_x)) + 1
            params.setModelStrcuture(hidden_layer_dim, SRF_depth, Tree_number,
                                     Use_Linear_Leaf, Use_Diverse_SRT)
            K = EPOCHS_SCSC * math.ceil(n_train / params.BATCH_SIZE)
            params.setLearningRate(1 / np.sqrt(K), 2 / np.sqrt(K))
            set_seed(42)
            model_srf = SRF(input_dim=d_x, depth=SRF_depth, output_dim=d_z,
                            tree_number=Tree_number, params=params).to(device).double()
            news_params = {'problem_name': 'Newsvendor', 'h_cost': h_cost_tensor,
                           'b_cost': b_cost_tensor, 'dimension': [d_x, d_y, d_z],
                           'sigma': sigma, 'n_m': n_m}
            # Generate each instance and solve its ERM and SAA oracle only once.
            data_cache = {}
            for instance in range(1, n_instances + 1):
                seed = 42 + instance - 1
                news_params['seed'] = seed
                set_seed(seed)
                data_gen = DataGenerator_News(n_train, n_test, news_params, device)
                X_train, Y_train, X_test, Y_test, train_loader = data_gen.get_data(params.BATCH_SIZE)
                F_star = calculate_F_star_saa_news(data_gen, X_test, news_params)
                erm_trainer = ERM_Trainer(copy.deepcopy(model_srf), train_loader, params, device, 'SRF')
                model_erm, _ = erm_trainer.train_erm(X_train, Y_train, news_params)
                F_erm = get_test_loss(model_erm, X_test, Y_test, news_params)
                data_cache[seed] = (X_train, Y_train, X_test, Y_test, train_loader, F_star, F_erm)
                print(f'Baseline seed={seed}: F_star={F_star:.8f}, F_erm={F_erm:.8f}', flush=True)

            # CWDRO and KLDRO have no epsilon; tune only their lambda.
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
                params.EPOCHS_SCSC = EPOCHS_SCSC if model_name == 'Causal-SDRO' else EPOCHS_KLDRO
                K = params.EPOCHS_SCSC * len(next(iter(data_cache.values()))[4])
                alpha_scale = alpha_scale_by_norm[norm] if model_name == 'Causal-SDRO' else np.nan
                beta_scale = beta_scale_fixed if model_name == 'Causal-SDRO' else np.nan
                learning_rate = (params.LR_KLDRO if model_name == 'KL-DRO' else
                                 0.09 if model_name == 'Causal-WDRO' else np.nan)
                for seed, data in data_cache.items():
                    X_train, Y_train, X_test, Y_test, train_loader, F_star, F_erm = data
                    news_params.update({'seed': seed, 'norm': norm, 'lambda': lam, 'epsilon': eps})
                    set_seed(seed)
                    start = time.perf_counter()
                    Q = np.nan
                    if model_name == 'Causal-SDRO':
                        params.setLearningRate(alpha_scale / np.sqrt(K), beta_scale / np.sqrt(K))
                        trainer = CSDRO_Trainer(copy.deepcopy(model_srf), train_loader, params, device, 'SRF')
                        model, _ = trainer.train_scsc(X_train, Y_train, news_params)
                        Q = trainer.Q
                    elif model_name == 'Causal-WDRO':
                        trainer = CWDRO_Trainer(copy.deepcopy(model_srf), train_loader, params, device, 'SRF')
                        model, _ = trainer.train_rtmlmc(X_train, Y_train, news_params)
                    else:
                        trainer = KLDRO_Trainer(copy.deepcopy(model_srf), train_loader, params, device, 'SRF')
                        model, _ = trainer.train_sgd(X_train, Y_train, news_params)
                    F_model = get_test_loss(model, X_test, Y_test, news_params)
                    gap = F_erm - F_star
                    P_raw = 100 * (F_erm - F_model) / gap if gap > 1e-10 else np.nan
                    results.append({'Model': model_name, 'norm': norm, 'lambda': lam, 'epsilon': eps,
                                    'alpha_scale': alpha_scale, 'beta_scale': beta_scale,
                                    'learning_rate': learning_rate,
                                    'K': K if model_name == 'Causal-SDRO' else trainer.epochs * (len(train_loader) if model_name == 'KL-DRO' else 1),
                                    'Q': Q, 'seed': seed, 'N': n_train, 'd_x': d_x, 'n_m': n_m,
                                    'F_star': F_star, 'F_erm': F_erm, 'F_model': F_model,
                                    'P_raw': P_raw, 'P': max(-100, P_raw) if np.isfinite(P_raw) else np.nan,
                                    'seconds': time.perf_counter() - start})
                    pd.DataFrame(results).to_csv(
                        os.path.join(OUTPUT_DIR, 'Newsvendor_All_CV_Details.csv'), index=False)
                    print(f'{len(results)} {model_name} p={norm} lambda={lam} epsilon={eps} '
                          f'n_m={n_m} alpha_scale={alpha_scale} beta_scale={beta_scale} seed={seed} '
                          f'P_raw={P_raw:.4f}', flush=True)

    save_cv_results(results)
