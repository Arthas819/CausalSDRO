"""
    This file provides cross validation for portfolio.
"""

import os
import sys
import copy
import math

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from Tools.Parameters import *
from Optimizers.Portfolio_Optimizer import *
from Optimizers.SDRO_Trainer import *
from Optimizers.CWDRO_Trainer import *
from Optimizers.KLDRO_Trainer import *

'''
    Changeable Parameters
'''
cv_csdro = True
cv_sdro = True
cv_cwdro = True
cv_kldro = True

# Select on 2021-2022 windows; check 2023 Q1 separately.
set_trade_dates = ['2021-01-03', '2021-04-01', '2021-07-01', '2021-10-01', '2022-01-03', '2022-04-01', '2022-07-01', '2022-10-03']
HISTORY_DAYS = 730
TRAIN_DAYS = 730
n_m_list = [1, 5, 10]
p_norm_list = [1, 2]
ETA_List = [5]
RANDOM_SEED = 42
RANDOM_SEEDS = [42, 43, 44]
STOCK_NUM = 50
d_x, d_y, d_z = 5, STOCK_NUM, STOCK_NUM + 1

# Only lambda and epsilon are searched here.
lambda_list = {
    'Causal-SDRO': [0.1, 1.0, 3.0, 10.0, 100.0],
    'SDRO': [0.1, 1.0, 3.0, 10.0, 100.0],
    'Causal-WDRO': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'KL-DRO': [0.01, 0.1, 1.0, 10.0]
}
epsilon_list = [0.01, 0.1, 0.5, 1.0, 2.0]

OUTPUT_DIR = os.path.join(PROJECT_DIR, 'Outputs', 'Outputs_Tuning')
output_path = os.path.join(OUTPUT_DIR, 'DRO_Portfolio_Best_Parameters.xlsx')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def reset_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_data(date, n_m, ETA, x_features, y_ret_select, params):
    x_row = x_features.loc[x_features.index < date].iloc[-HISTORY_DAYS:].tail(TRAIN_DAYS)
    y_row = y_ret_select.loc[y_ret_select.index < date].iloc[-HISTORY_DAYS:].tail(TRAIN_DAYS)
    mean, std = x_row.mean(), x_row.std() + 1e-8
    X_train = torch.tensor(((x_row - mean) / std).values, device=device, dtype=torch.float64)
    Y_train = torch.tensor(y_row.values, device=device, dtype=torch.float64)
    x_today = x_features.loc[date:].iloc[:1]
    X_test = torch.tensor(((x_today - mean) / std).values, device=device, dtype=torch.float64)
    X_train, Y_train, group_index = group_portfolio_data(X_train, Y_train, n_m)
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=params.BATCH_SIZE, shuffle=False)
    future_return = y_ret_select.loc[date:].head(60)
    pf_params = {
        'problem_name': 'Portfolio', 'eta': ETA, 'historical_days': HISTORY_DAYS, 'train_days': TRAIN_DAYS,
        'n_m': n_m, 'group_index': group_index, 'dimension': [d_x, d_y, d_z],
        'B': params.B, 'a': params.a, 'z0_bound': params.z0_bound,
        'norm': 2, 'lambda': 1.0, 'epsilon': 1.0, 'future_return': future_return
    }
    w_star = post_hc_solver(future_return.values, ETA)
    F_star = metrics((future_return.values * w_star).sum(axis=1), ETA)[0]
    return X_train, Y_train, X_test, train_loader, pf_params, F_star


if __name__ == '__main__':
    # Reuse the selected algorithm settings; no matching by N, d_x or n_m.
    default_parameters = pd.read_excel(output_path, sheet_name='Best_Parameters')
    params = Parameters_Portfolio()
    params.setModelStrcuture(250, math.ceil(math.log2(d_x)) + 1, 20, False, False)
    all_data = pd.read_csv(os.path.join(PROJECT_DIR, 'DRO_Portfolio', 'data',
                                      '[Return and Features]20170101-20230331.csv'), index_col=0)
    x_features = all_data.iloc[:, -5:]
    y_returns = all_data.iloc[:, :-5]
    selected_cols = np.random.RandomState(RANDOM_SEED).choice(y_returns.columns, STOCK_NUM, replace=False)
    y_ret_select = y_returns[selected_cols]

    models = []
    for enabled, name in [(cv_csdro, 'Causal-SDRO'), (cv_sdro, 'SDRO'),
                          (cv_cwdro, 'Causal-WDRO'), (cv_kldro, 'KL-DRO')]:
        if enabled:
            models.append(name)

    # Cache data and ERM once for each validation instance.
    data_cache = {}
    erm_records = []
    erm_settings = default_parameters[default_parameters['Model'] == 'ERM'].iloc[0]
    for ETA in ETA_List:
        for n_m in n_m_list:
            for date in set_trade_dates:
                X_train, Y_train, X_test, train_loader, pf_params, F_star = get_data(
                    date, n_m, ETA, x_features, y_ret_select, params)
                for seed in RANDOM_SEEDS:
                    params.setCVParameters(erm_settings, len(train_loader))
                    reset_seed(seed)
                    model_srf = SRF(d_x, params.SRF_depth, d_z, params.Tree_number, params).to(device).double()
                    trainer = ERM_Trainer(copy.deepcopy(model_srf), train_loader, params, device, 'SRF')
                    model_erm, _ = trainer.train_erm(X_train, Y_train, pf_params)
                    F_erm = get_test_loss(model_erm, X_test, None, pf_params)
                    if F_erm <= F_star:
                        raise ValueError('The ERM loss must exceed the post-hoc loss when calculating P.')
                    data_cache[ETA, n_m, date, seed] = (
                        X_train, Y_train, X_test, train_loader, pf_params, F_star, F_erm, model_srf)
                    erm_records.append({'Model': 'ERM', 'norm': 0, 'ETA': ETA, 'n_m': n_m,
                                        'date': date, 'seed': seed, 'N': len(X_train), 'M': len(X_train) // n_m,
                                        'F_star': F_star, 'F_erm': F_erm, 'F_model': F_erm, 'P': 0.0})

    results = []
    for model_name in models:
        for norm in ([0] if model_name == 'KL-DRO' else p_norm_list):
            settings = default_parameters[
                (default_parameters['Model'] == model_name) & (default_parameters['norm'] == norm)].iloc[0]
            eps_values = epsilon_list if model_name in ('Causal-SDRO', 'SDRO') else [0.0]
            for lambda_val in lambda_list[model_name]:
                for epsilon_val in eps_values:
                    for (ETA, n_m, date, seed), data in data_cache.items():
                        X_train, Y_train, X_test, train_loader, base_params, F_star, F_erm, model_srf = data
                        pf_params = dict(base_params, norm=norm, epsilon=epsilon_val)
                        pf_params['lambda'] = lambda_val
                        params.setCVParameters(settings, len(train_loader))
                        reset_seed(seed)
                        model = SRF(d_x, params.SRF_depth, d_z, params.Tree_number, params).to(device).double()
                        if model_name == 'Causal-SDRO':
                            trainer = CSDRO_Trainer(model, train_loader, params, device, 'SRF')
                            trained, _ = trainer.train_scsc(X_train, Y_train, pf_params)
                        elif model_name == 'SDRO':
                            trainer = SDRO_Trainer(model, train_loader, params, device, 'SRF')
                            trained, _ = trainer.train_scsc(X_train, Y_train, pf_params)
                        elif model_name == 'Causal-WDRO':
                            trainer = CWDRO_Trainer(model, train_loader, params, device, 'SRF')
                            trained, _ = trainer.train_rtmlmc(X_train, Y_train, pf_params)
                        else:
                            trainer = KLDRO_Trainer(model, train_loader, params, device, 'SRF')
                            trained, _ = trainer.train_sgd(X_train, Y_train, pf_params)
                        F_model = get_test_loss(trained, X_test, None, pf_params)
                        P = 100 * (F_erm - F_model) / (F_erm - F_star)
                        results.append({
                            'Model': model_name, 'norm': norm, 'ETA': ETA, 'n_m': n_m, 'date': date, 'seed': seed,
                            'N': len(X_train), 'M': len(X_train) // n_m,
                            'lambda': lambda_val, 'epsilon': epsilon_val,
                            'F_star': F_star, 'F_erm': F_erm, 'F_model': F_model, 'P': P
                        })
                        print(model_name, norm, date, n_m, lambda_val, epsilon_val, 'P=', P)

    df_results = pd.DataFrame(results)
    df_results['Year'] = pd.to_datetime(df_results['date']).dt.year
    group_columns = ['Model', 'norm', 'ETA', 'lambda', 'epsilon']
    df_summary = df_results.groupby(group_columns)[['F_star', 'F_erm', 'F_model', 'P']].mean().reset_index()
    # Prefer the better result in the worse validation year.
    yearly_P = df_results.groupby(group_columns + ['Year'])['P'].mean()
    worst_year_P = yearly_P.groupby(level=group_columns).min().rename('P_min_year').reset_index()
    df_summary = df_summary.merge(worst_year_P, on=group_columns)
    best_parameters = []
    for (model_name, norm, ETA), table in df_summary.groupby(['Model', 'norm', 'ETA']):
        # SDRO shares lambda and epsilon with Causal-SDRO.
        if model_name == 'SDRO':
            causal = next((row for row in best_parameters if row['Model'] == 'Causal-SDRO'
                           and row['norm'] == norm and row['ETA'] == ETA), None)
            if causal is None:
                causal = default_parameters[(default_parameters['Model'] == 'Causal-SDRO')
                                            & (default_parameters['norm'] == norm)].iloc[0]
            table = table[(table['lambda'] == causal['best_lambda'])
                          & (table['epsilon'] == causal['best_epsilon'])]
        best = table.sort_values(['P_min_year', 'P'], ascending=False).iloc[0]
        settings = default_parameters[
            (default_parameters['Model'] == model_name) & (default_parameters['norm'] == norm)].iloc[0]
        row = {key: settings[key] for key in
               ['Model', 'norm', 'K', 'Q']}
        row.update(ETA=ETA, N=TRAIN_DAYS, best_lambda=best['lambda'], best_epsilon=best['epsilon'], best_P=best['P'])
        best_parameters.append(row)
    erm_row = {key: erm_settings[key] for key in ['Model', 'norm', 'ETA', 'N', 'best_lambda', 'best_epsilon', 'K', 'Q']}
    erm_row['best_P'] = 0.0
    best_parameters.append(erm_row)

    # Plain output tables, with no custom formatting.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        pd.DataFrame(best_parameters).to_excel(writer, sheet_name='Best_Parameters', index=False)
        df_summary.to_excel(writer, sheet_name='CV_Summary', index=False)
        pd.concat([df_results, pd.DataFrame(erm_records)], ignore_index=True).to_excel(
            writer, sheet_name='CV_Results', index=False)
    print('Saved:', output_path)

