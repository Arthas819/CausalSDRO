"""
    This file solves all compared DRO models.
"""

import copy
import os
import pandas as pd

from Tools.Data_Generator_Newsvendor import *
from Tools.Data_Generator_Inventory import *
from Tools.Decision_Rules import *

from Optimizers.ERM_Trainer import *
from Optimizers.Smooth_Trainer import *
from Optimizers.CSDRO_Trainer import *
from Optimizers.SDRO_Trainer import *
from Optimizers.CWDRO_Trainer import *
from Optimizers.KLDRO_Trainer import *

'''
    Solve all DRO models
'''
class Compared_DRO_Optimizer:

    def __init__(self, F_star, n_train, n_test, instance, fixed_params, pro_params, device):

        self.F_star = F_star
        self.n_train = n_train
        self.n_test = n_test
        self.instance = instance # For portfolio, instance=date
        self.params = fixed_params
        self.baseline_epochs = fixed_params.EPOCHS_SCSC
        self.pro_params = pro_params
        self.device = device
        self.problem = self.pro_params['problem_name']

        self.d_x = pro_params['dimension'][0]
        self.d_z = pro_params['dimension'][2]
        self.SRF_depth = self.params.SRF_depth
        self.Tree_number = self.params.Tree_number

        # This file saves best parameters for all DRO models.
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.best_parameters = pd.read_excel(os.path.join(project_dir, "Outputs/Outputs_Tuning",
                                                          f"DRO_{self.problem}_Best_Parameters.xlsx"))

        # Only for portfolio
        self.RANDOM_SEED = 42
        self.HISTORY_DAYS = pro_params.get('historical_days', 730)
        self.TRAIN_DAYS = pro_params.get('train_days', 730)
        self.STOCK_NUM = 50

    def configure_csdro(self, best_row, train_loader):
        self.pro_params['lambda'] = float(best_row['best_lambda'])
        self.pro_params['epsilon'] = float(best_row['best_epsilon'])
        if self.problem == 'Newsvendor':
            self.params.EPOCHS_SCSC = math.ceil(best_row['K'] / len(train_loader))
            K = self.params.EPOCHS_SCSC * len(train_loader)
            self.params.setLearningRate(best_row.get('best_alpha_scale', 1.0) / math.sqrt(K),
                                         best_row.get('best_beta_scale', 2.0) / math.sqrt(K))
            self.params.Q = int(best_row.get('Q', math.ceil(K ** 0.25)))
        elif self.problem == 'Inventory':
            self.params.EPOCHS_SCSC = self.baseline_epochs
            self.params.setCSDROParameters(len(train_loader), best_row)

    def solve_all_models(self):
        data_gen = None
        # Generate data
        if self.problem == 'Newsvendor':
            set_seed(self.pro_params.get('seed', 42))
            data_gen = DataGenerator_News(self.n_train, self.n_test, self.pro_params, self.device)
            # Get normalized data
            X_train, Y_train, X_test, Y_test, train_loader = data_gen.get_data(self.params.BATCH_SIZE)
            self.F_star = calculate_F_star_saa_news(data_gen, X_test, self.pro_params)
        elif self.problem == 'Inventory':
            set_seed(self.pro_params.get('seed', 42))
            data_gen = DataGenerator_Inventory(self.n_train, self.n_test, self.pro_params, self.device)
            # Get normalized data
            X_train, Y_train, X_test, Y_test, train_loader = data_gen.get_data(self.params.BATCH_SIZE)
            self.F_star = calculate_F_star_theoretical_is(data_gen, X_test, Y_test, self.pro_params)
        elif self.problem == 'Portfolio':
            # Read data
            all_data = pd.read_csv('../DRO_Portfolio/data/[Return and Features]20170101-20230331.csv')
            all_data = all_data.set_index(all_data.columns[0])
            y_returns = all_data.iloc[:, :-5]
            x_features = all_data.iloc[:, -5:]
            # Select Stocks Randomly
            rng = np.random.RandomState(self.RANDOM_SEED)
            selected_cols = rng.choice(y_returns.columns, size=self.STOCK_NUM, replace=False)
            y_ret_select = y_returns[selected_cols]
            # History features
            x_historical_1 = x_features.loc[x_features.index < self.instance, :]
            x_historical_row = x_historical_1.iloc[-self.HISTORY_DAYS:, :].tail(self.TRAIN_DAYS)
            # Normalization
            scaler_mean = x_historical_row.mean()
            scaler_std = x_historical_row.std()
            x_historical = (x_historical_row - scaler_mean) / (scaler_std + 1e-8)
            # Feature Today
            x_today_row = x_features.loc[self.instance:, :].iloc[:1, :]
            # Normalization (using mean and std of training set)
            x_today = (x_today_row - scaler_mean) / (scaler_std + 1e-8)
            # History return
            y_historical_1 = y_ret_select.loc[y_ret_select.index < self.instance, :]
            y_historical = y_historical_1.iloc[-self.HISTORY_DAYS:, :].tail(self.TRAIN_DAYS)
            # Return today
            y_today = y_ret_select.loc[self.instance:, :].iloc[:1, :]
            # Transform to Tensor. Training and testing sets
            X_train = torch.tensor(x_historical.values, dtype=torch.float64).to(self.device)
            Y_train = torch.tensor(y_historical.values, dtype=torch.float64).to(self.device)
            X_test = torch.tensor(x_today.values, dtype=torch.float64).to(self.device)
            Y_test = torch.tensor(y_today.values, dtype=torch.float64).to(self.device)
            # Disjoint kNN groups, shared by all methods.
            n_m = self.pro_params['n_m']
            X_train, Y_train, group_index = group_portfolio_data(X_train, Y_train, n_m)
            self.pro_params['group_index'] = group_index
            self.n_train = len(X_train)
            print(f"Groups: n_m={n_m}, M={self.n_train // n_m}, N={self.n_train}")
            # Set train loader
            train_dataset = TensorDataset(X_train, Y_train)
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=self.params.BATCH_SIZE,
                shuffle=False
            )
            K = self.params.EPOCHS_SCSC * len(train_loader)
            self.params.setLearningRate(1 / np.sqrt(K), 2 / np.sqrt(K))
        else:
            X_train = Y_train = X_test = Y_test = train_loader = None

        if self.problem in ('Newsvendor', 'Inventory'):
            X_validation, Y_validation = validation_data(
                data_gen, getattr(self.params, 'WARM_VALIDATION_SIZE', 1000))
        warm_diagnostics = {}

        # Create models, use .double() to set torch.float64
        if self.params.Use_Linear_Leaf == True:
            model_name = 'SRF(Linear Leaf)'
        else:
            model_name = 'SRF(Constant Leaf)'

        # Match the CV initialization for every method.
        if self.problem in ('Newsvendor', 'Portfolio'):
            set_seed(42)
        elif self.problem == 'Inventory':
            set_seed(self.pro_params.get('seed', 42))
        # SRF model
        model_srf = SRF(input_dim=self.d_x, depth=self.SRF_depth, output_dim=self.d_z,
                        tree_number=self.Tree_number, params=self.params).to(self.device).double()

        # Train this model as ERM, return a trained model and a historical loss list for drawing convergence curve
        if self.problem in ('Newsvendor', 'Inventory'):
            set_seed(self.pro_params.get('seed', 42))
        if self.problem == 'Portfolio':
            erm_row = self.best_parameters[self.best_parameters['Model'] == 'ERM'].iloc[0]
            self.params.setCVParameters(erm_row, len(train_loader))
        erm_trainer = ERM_Trainer(copy.deepcopy(model_srf), train_loader, self.params, self.device, 'SRF')
        model_erm, history_saa = erm_trainer.train_erm(X_train, Y_train, self.pro_params)
        F_erm = get_test_loss(model_erm, X_test, Y_test, self.pro_params)

        # Solve Optimal-transport-based DRO, each returns a trained model and a historical loss list
        norm_list = [1, 2]
        F_csdro   = [0, 0]
        F_sdro    = [0, 0]
        F_cwdro   = [0, 0]
        F_smooth  = [0, 0]
        epsilon_smooth = [0, 0]

        for norm in norm_list:
            # Set p-norm
            self.pro_params['norm'] = norm

            # Solve p-Causal-SDRO
            best_row = self.best_parameters[(self.best_parameters['Model'] == 'Causal-SDRO') & (self.best_parameters['norm'] == norm)]
            if self.problem in ('Newsvendor', 'Inventory'):
                self.configure_csdro(best_row.iloc[0], train_loader)
                model_smooth, model_csdro, diagnostics = fit_smooth_csdro(
                    model_srf, train_loader, self.params, self.device, X_train, Y_train,
                    X_validation, Y_validation, self.pro_params.copy())
                warm_diagnostics.update(diagnostics)
                epsilon_smooth[norm-1] = self.pro_params['epsilon']
                F_smooth[norm-1] = evaluate_loss(model_smooth, X_test, Y_test, self.pro_params)
                F_csdro[norm-1] = evaluate_loss(model_csdro, X_test, Y_test, self.pro_params)
            else:
                self.pro_params['lambda'] = best_row.iloc[0]['best_lambda']
                self.pro_params['epsilon'] = best_row.iloc[0]['best_epsilon']
                self.params.setCVParameters(best_row.iloc[0], len(train_loader))
                set_seed(42)
                model_srf = SRF(self.d_x, self.SRF_depth, self.d_z, self.Tree_number,
                                self.params).to(self.device).double()
                csdro_trainer = CSDRO_Trainer(copy.deepcopy(model_srf), train_loader,
                                             self.params, self.device, 'SRF')
                model_csdro, history_csdro = csdro_trainer.train_scsc(X_train, Y_train, self.pro_params)
                F_csdro[norm-1] = get_test_loss(model_csdro, X_test, Y_test, self.pro_params)

            # Newsvendor SDRO reuses the Causal-SDRO parameters.
            parameter_model = 'Causal-SDRO' if self.problem in ('Newsvendor', 'Inventory') else 'SDRO'
            best_row = self.best_parameters[(self.best_parameters['Model'] == parameter_model) & (self.best_parameters['norm'] == norm)]
            self.pro_params['lambda'] = best_row.iloc[0]['best_lambda']
            self.pro_params['epsilon'] = best_row.iloc[0]['best_epsilon']
            if self.problem == 'Newsvendor':
                set_seed(self.pro_params.get('seed', 42))
            if self.problem == 'Portfolio':
                self.params.setCVParameters(best_row.iloc[0], len(train_loader))
                set_seed(42)
                model_srf = SRF(self.d_x, self.SRF_depth, self.d_z, self.Tree_number, self.params).to(self.device).double()
            if self.problem == 'Inventory':
                set_seed(self.pro_params.get('seed', 42))
            if self.problem in ('Newsvendor', 'Inventory') and self.pro_params['n_m'] == 1:
                # Identical objectives at n_m=1: reuse the same fitted policy and loss.
                F_sdro[norm-1] = F_csdro[norm-1]
            else:
                sdro_trainer = SDRO_Trainer(copy.deepcopy(model_srf), train_loader, self.params, self.device, 'SRF')
                model_sdro, history_sdro = sdro_trainer.train_scsc(X_train, Y_train, self.pro_params)
                F_sdro[norm-1] = get_test_loss(model_sdro, X_test, Y_test, self.pro_params)

            # Solve p-Causal-WDRO
            best_row = self.best_parameters[(self.best_parameters['Model'] == 'Causal-WDRO') & (self.best_parameters['norm'] == norm)]
            self.pro_params['lambda'] = best_row.iloc[0]['best_lambda']
            self.pro_params['epsilon'] = best_row.iloc[0]['best_epsilon']
            if self.problem == 'Newsvendor':
                set_seed(self.pro_params.get('seed', 42))
            if self.problem == 'Portfolio':
                self.params.setCVParameters(best_row.iloc[0], len(train_loader))
                set_seed(42)
                model_srf = SRF(self.d_x, self.SRF_depth, self.d_z, self.Tree_number, self.params).to(self.device).double()
            if self.problem == 'Inventory':
                set_seed(self.pro_params.get('seed', 42))
            cwdro_trainer = CWDRO_Trainer(copy.deepcopy(model_srf), train_loader, self.params, self.device, 'SRF')
            model_cwdro, history_cwdro = cwdro_trainer.train_rtmlmc(X_train, Y_train, self.pro_params)
            F_cwdro[norm-1] = get_test_loss(model_cwdro, X_test, Y_test, self.pro_params)

        # Solve KL-DRO
        best_row = self.best_parameters[
            (self.best_parameters['Model'] == 'KL-DRO')]
        self.pro_params['lambda'] = best_row.iloc[0]['best_lambda']
        self.pro_params['epsilon'] = best_row.iloc[0]['best_epsilon']
        if self.problem == 'Newsvendor':
            self.params.EPOCHS_SCSC = math.ceil(best_row.iloc[0]['K'] / len(train_loader))
            set_seed(self.pro_params.get('seed', 42))
        if self.problem == 'Portfolio':
            self.params.setCVParameters(best_row.iloc[0], len(train_loader))
            set_seed(42)
            model_srf = SRF(self.d_x, self.SRF_depth, self.d_z, self.Tree_number, self.params).to(self.device).double()
        if self.problem == 'Inventory':
            self.params.EPOCHS_SCSC = self.baseline_epochs
            set_seed(self.pro_params.get('seed', 42))
        kldro_trainer = KLDRO_Trainer(copy.deepcopy(model_srf), train_loader, self.params, self.device, 'SRF')
        model_kldro, history_kldro = kldro_trainer.train_sgd(X_train, Y_train, self.pro_params)
        F_kldro = get_test_loss(model_kldro, X_test, Y_test, self.pro_params)

        # To calculate Out-of-sample performance (coefficient of prescriptiveness) (%)
        erm_gap = F_erm - self.F_star
        if self.problem in ('Newsvendor', 'Inventory') and erm_gap <= 1e-10:
            erm_gap = np.nan

        # Record Parameters
        record = None
        if self.problem == 'Newsvendor':
            record = "h=" + str(self.pro_params['h_cost'].item()) + ", b=" + str(self.pro_params['b_cost'].item())
        elif self.problem == 'Portfolio':
            record = self.pro_params['eta']
        elif self.problem == 'Inventory':
            record = "S=" + str(self.pro_params['s_cost'].cpu().numpy().tolist())

        result_list = {
                       'Model': model_name,
                       'Parameters': record,
                       'N': self.n_train, 'D_X': self.d_x, 'Instance': self.instance,
                       'F*': self.F_star, 'F_ERM': F_erm,
                       'F_1_CSDRO': F_csdro[0], 'F_2_CSDRO': F_csdro[1], 'F_KLDRO': F_kldro,
                       'F_1_SDRO': F_sdro[0], 'F_2_SDRO': F_sdro[1],
                       'F_1_CWDRO': F_cwdro[0], 'F_2_CWDRO': F_cwdro[1],
                       'P_1-Causal-SDRO': max(-100, (1 - (F_csdro[0] - self.F_star) / erm_gap) * 100), 'P_2-Causal-SDRO': max(-100, (1 - (F_csdro[1] - self.F_star) / erm_gap) * 100) ,
                       'P_1-SDRO': max(-100, (1 - (F_sdro[0] - self.F_star) / erm_gap) * 100), 'P_2-SDRO': max(-100, (1 - (F_sdro[1] - self.F_star) / erm_gap) * 100 ) ,
                       'P_1-Causal-WDRO': max(-100, (1 - (F_cwdro[0] - self.F_star) / erm_gap) * 100), 'P_2-Causal-WDRO': max(-100, (1 - (F_cwdro[1] - self.F_star) / erm_gap) * 100 ),
                       'P_KL-DRO': max(-100, (1 - (F_kldro - self.F_star) / erm_gap) * 100)
                    }

        if self.problem in ('Newsvendor', 'Inventory'):
            result_list.update({
                'F_1_Smooth': F_smooth[0], 'F_2_Smooth': F_smooth[1],
                'epsilon_1_Smooth': epsilon_smooth[0], 'epsilon_2_Smooth': epsilon_smooth[1],
                'P_1-Smooth': max(-100, (1 - (F_smooth[0] - self.F_star) / erm_gap) * 100),
                'P_2-Smooth': max(-100, (1 - (F_smooth[1] - self.F_star) / erm_gap) * 100)
            })

        if self.problem in ('Newsvendor', 'Inventory'):
            losses = {'1-Causal-SDRO': F_csdro[0], '2-Causal-SDRO': F_csdro[1],
                      '1-SDRO': F_sdro[0], '2-SDRO': F_sdro[1],
                      '1-Causal-WDRO': F_cwdro[0], '2-Causal-WDRO': F_cwdro[1], 'KL-DRO': F_kldro,
                      '1-Smooth': F_smooth[0], '2-Smooth': F_smooth[1]}
            for method, value in losses.items():
                P_raw = 100 * (F_erm - value) / erm_gap
                result_list[f'P_{method}'] = max(-100, P_raw) if np.isfinite(P_raw) else np.nan

        # Report the original Portfolio P without clipping.
        if self.problem == 'Portfolio':
            losses = {'1-Causal-SDRO': F_csdro[0], '2-Causal-SDRO': F_csdro[1],
                      '1-SDRO': F_sdro[0], '2-SDRO': F_sdro[1],
                      '1-Causal-WDRO': F_cwdro[0], '2-Causal-WDRO': F_cwdro[1], 'KL-DRO': F_kldro}
            for method, value in losses.items():
                result_list[f'P_{method}'] = 100 * (F_erm - value) / erm_gap

        result_list.update(warm_diagnostics)

        if self.problem in ('Newsvendor', 'Inventory', 'Portfolio'):
            result_list.update({'n_m': self.pro_params['n_m'],
                                'M': self.n_train // self.pro_params['n_m']})

        return result_list
