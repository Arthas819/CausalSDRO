"""
    This file solves all compared DRO models.
"""

import copy
import pandas as pd

from CausalSDRO.Tools.Data_Generator_Newsvendor import *
from CausalSDRO.Tools.Data_Generator_Inventory import *
from CausalSDRO.Tools.Decision_Rules import *

from CausalSDRO.Optimizers.ERM_Trainer import *
from CausalSDRO.Optimizers.CSDRO_Trainer_Revise import *
from CausalSDRO.Optimizers.SDRO_Trainer import *
from CausalSDRO.Optimizers.CWDRO_Trainer import *
from CausalSDRO.Optimizers.KLDRO_Trainer import *

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
        self.pro_params = pro_params
        self.device = device
        self.problem = self.pro_params['problem_name']

        self.d_x = pro_params['dimension'][0]
        self.d_z = pro_params['dimension'][2]
        self.SRF_depth = self.params.SRF_depth
        self.Tree_number = self.params.Tree_number

        # This file saves best parameters for all DRO models.
        self.best_parameters = pd.read_excel(f"../Outputs/Outputs_Tuning/"
                                             f"DRO_{self.problem}_Best_Parameters.xlsx")

        # Only for portfolio
        self.RANDOM_SEED = 42
        self.HISTORY_DAYS = 730
        self.STOCK_NUM = 50

    def solve_all_models(self):
        data_gen = None
        # Generate data
        if self.problem == 'Newsvendor':
            data_gen = DataGenerator_News(self.n_train, self.n_test, self.pro_params, self.device)
            # Get normalized data
            X_train, Y_train, X_test, Y_test, train_loader = data_gen.get_data(self.params.BATCH_SIZE)
        elif self.problem == 'Inventory':
            data_gen = DataGenerator_Inventory(self.n_train, self.n_test, self.pro_params, self.device)
            # Get normalized data
            X_train, Y_train, X_test, Y_test, train_loader = data_gen.get_data(self.params.BATCH_SIZE)
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
            x_historical_1 = x_features.loc[:self.instance, :]
            x_historical_row = x_historical_1.iloc[-self.HISTORY_DAYS:, :]
            # Normalization
            scaler_mean = x_historical_row.mean()
            scaler_std = x_historical_row.std()
            x_historical = (x_historical_row - scaler_mean) / (scaler_std + 1e-8)
            # Feature Today
            x_today_row = x_features.loc[self.instance:, :].iloc[:1, :]
            # Normalization (using mean and std of training set)
            x_today = (x_today_row - scaler_mean) / (scaler_std + 1e-8)
            # History return
            y_historical_1 = y_ret_select.loc[:self.instance, :]
            y_historical = y_historical_1.iloc[-self.HISTORY_DAYS:, :]
            # Return today
            y_today = y_ret_select.loc[self.instance:, :].iloc[:1, :]
            # Transform to Tensor. Training and testing sets
            X_train = torch.tensor(x_historical.values, dtype=torch.float64).to(self.device)
            Y_train = torch.tensor(y_historical.values, dtype=torch.float64).to(self.device)
            X_test = torch.tensor(x_today.values, dtype=torch.float64).to(self.device)
            Y_test = torch.tensor(y_today.values, dtype=torch.float64).to(self.device)
            # Set train loader
            train_dataset = TensorDataset(X_train, Y_train)
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=self.params.BATCH_SIZE,
                shuffle=False
            )
        else:
            X_train = Y_train = X_test = Y_test = train_loader = None

        # Create models, use .double() to set torch.float64
        if self.params.Use_Linear_Leaf == True:
            model_name = 'SRF(Linear Leaf)'
        else:
            model_name = 'SRF(Constant Leaf)'

        # SRF model
        model_srf = SRF(input_dim=self.d_x, depth=self.SRF_depth, output_dim=self.d_z,
                        tree_number=self.Tree_number, params=self.params).to(self.device).double()

        # Train this model as ERM, return a trained model and a historical loss list for drawing convergence curve
        erm_trainer = ERM_Trainer(copy.deepcopy(model_srf), train_loader, self.params, self.device, 'SRF')
        model_erm, history_saa = erm_trainer.train_erm(X_train, Y_train, self.pro_params)
        F_erm = get_test_loss(model_erm, X_test, Y_test, self.pro_params)

        # Solve Optimal-transport-based DRO, each returns a trained model and a historical loss list
        norm_list = [1, 2]
        F_csdro   = [0, 0]
        F_sdro    = [0, 0]
        F_cwdro   = [0, 0]

        for norm in norm_list:
            # Set p-norm
            self.pro_params['norm'] = norm

            # Solve p-Causal-SDRO
            best_row = self.best_parameters[(self.best_parameters['Model'] == 'Causal-SDRO') & (self.best_parameters['norm'] == norm)]
            self.pro_params['lambda'] = best_row.iloc[0]['best_lambda']
            self.pro_params['epsilon'] = best_row.iloc[0]['best_epsilon']
            csdro_trainer = CSDRO_Trainer(copy.deepcopy(model_srf), train_loader, self.params, self.device, 'SRF')
            model_csdro, history_csdro = csdro_trainer.train_scsc(X_train, Y_train, self.pro_params)
            F_csdro[norm-1] = get_test_loss(model_csdro, X_test, Y_test, self.pro_params)

            # Solve p-SDRO
            best_row = self.best_parameters[(self.best_parameters['Model'] == 'SDRO') & (self.best_parameters['norm'] == norm)]
            self.pro_params['lambda'] = best_row.iloc[0]['best_lambda']
            self.pro_params['epsilon'] = best_row.iloc[0]['best_epsilon']
            sdro_trainer = SDRO_Trainer(copy.deepcopy(model_srf), train_loader, self.params, self.device, 'SRF')
            model_sdro, history_sdro = sdro_trainer.train_scsc(X_train, Y_train, self.pro_params)
            F_sdro[norm-1] = get_test_loss(model_sdro, X_test, Y_test, self.pro_params)

            # Solve p-Causal-WDRO
            best_row = self.best_parameters[(self.best_parameters['Model'] == 'Causal-WDRO') & (self.best_parameters['norm'] == norm)]
            self.pro_params['lambda'] = best_row.iloc[0]['best_lambda']
            self.pro_params['epsilon'] = best_row.iloc[0]['best_epsilon']
            cwdro_trainer = CWDRO_Trainer(copy.deepcopy(model_srf), train_loader, self.params, self.device, 'SRF')
            model_cwdro, history_cwdro = cwdro_trainer.train_rtmlmc(X_train, Y_train, self.pro_params)
            F_cwdro[norm-1] = get_test_loss(model_cwdro, X_test, Y_test, self.pro_params)

        # Solve KL-DRO
        best_row = self.best_parameters[
            (self.best_parameters['Model'] == 'KL-DRO')]
        self.pro_params['lambda'] = best_row.iloc[0]['best_lambda']
        self.pro_params['epsilon'] = best_row.iloc[0]['best_epsilon']
        kldro_trainer = KLDRO_Trainer(copy.deepcopy(model_srf), train_loader, self.params, self.device, 'SRF')
        model_kldro, history_kldro = kldro_trainer.train_sgd(X_train, Y_train, self.pro_params)
        F_kldro = get_test_loss(model_kldro, X_test, Y_test, self.pro_params)

        # To calculate Out-of-sample performance (coefficient of prescriptiveness) (%)
        erm_gap = F_erm - self.F_star

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

        return result_list
