"""
    This file Solves Causal-WDRO model as a
    contextual stochastic bilevel optimization.
"""

from tqdm import tqdm
from CausalSDRO.Tools.Functions import *
from torch.autograd import Variable

class CWDRO_Trainer:
    def __init__(self, model, train_loader, params, device, model_type):
        # Basic settings
        self.model =  model.to(device).double()
        self.model_type = model_type
        self.train_loader = train_loader
        self.params = params
        self.device = device
        # self.epochs = params.EPOCHS_SCSC
        self.epochs = 100
        self.batch_size = params.BATCH_SIZE

        # A list to store loss history
        self.train_loss_history = []

    def train_rtmlmc(self, X_train, Y_train, pro_params):
        self.n_train = X_train.shape[0]
        self.d_x = X_train.shape[1]
        self.d_y = Y_train.shape[1]
        self.d_z = pro_params['dimension'][2]

        self.pro_params = pro_params
        self.lam = pro_params['lambda']
        self.beta = 5
        self.y_step_size0 = 1e-2
        self.m = nn.Softplus(beta=self.beta)

        p = 0.5
        K_sample_max = 5
        # sampling from truncated gemoetric distribution
        self.elements = np.arange(K_sample_max) + 1
        probabilities = p ** (self.elements - 1)
        self.probabilities = probabilities / np.sum(probabilities)

        # Basic settings
        acc_test_hist = []
        theta_step_size0 = 0.09

        self.model.train()
        optimizer_theta = torch.optim.Adam(self.model.parameters(), amsgrad = True, lr=theta_step_size0)

        pbar_rtmlmc = tqdm(range(self.epochs), desc=f"Training Causal-WDRO ({self.model_type})", leave=False)

        # Update parameters by RT-MLMC
        for epoch in pbar_rtmlmc:
            batch_size_inner = 2
            optimizer_theta.zero_grad()
            loss_adv_list = [self.RTMLMC_obj_oracle(X_train, Y_train, self.batch_size) for i in range(batch_size_inner)]
            loss_adv_mean   = torch.mean(torch.stack(loss_adv_list), dim=0)
            loss_adv_mean.backward()
            optimizer_theta.step()

            if (epoch + 1) % 5 == 0:

                self.model.eval()

                with torch.no_grad():
                    # Input full dataset
                    eval_loss = self.get_eval_loss(self.model, X_train, Y_train)

                # Record historical loss
                self.train_loss_history.append(eval_loss.item())
                pbar_rtmlmc.set_postfix(TestDRO=eval_loss.item())

        return self.model, self.train_loss_history

    def RTMLMC_obj_oracle(self, x_Tr, z_Tr, batch_size_z):
        # Input: X_train, Y_train, Batch_size
        K_sample = int(np.random.choice(list(self.elements), 1, list(self.probabilities)))
        x_Tr_size, _ = x_Tr.shape
        idx = np.random.randint(x_Tr_size)

        x_Tr_idx = x_Tr[idx:idx + 1, :]
        z_Tr_idx = z_Tr[idx:idx + 1, :]

        train_data_z_Tr = torch.utils.data.TensorDataset(z_Tr_idx.float())
        train_data_loader_z = torch.utils.data.DataLoader(train_data_z_Tr, batch_size=batch_size_z, shuffle=True)
        y_hat0 = torch.zeros_like(x_Tr_idx)
        y_10, y_K0, y_K10 = self.epoch_SGD_revision(train_data_loader_z, x_Tr_idx, y_hat0, self.model,
                                                                    K_sample, self.lam, self.y_step_size0, self.beta, z_Tr_idx)
        z_Tr_j = next(iter(train_data_loader_z))[0]
        z_Tr_j = Variable(to_tensor(z_Tr_j), requires_grad=False)

        logit_y10, logit_yK0, logit_yK10 = self.model(y_10), self.model(y_K0), self.model(y_K10)

        loss_y10  = get_loss(logit_y10,  z_Tr_j, self.pro_params)
        loss_yK0  = get_loss(logit_yK0,  z_Tr_j, self.pro_params)
        loss_yK10 = get_loss(logit_yK10, z_Tr_j, self.pro_params)

        loss_adv = torch.mean(loss_y10) + 1/self.probabilities[K_sample-1] * (torch.mean(loss_yK10) - torch.mean(loss_yK0))
        return loss_adv

    def epoch_SGD_revision(self, train_data_loader, x_Tr, y_hat0, model, K_sample, Lambda, y_step_size0,
                           beta0, z_Tr_idx):

        y_hat = Variable(to_tensor(y_hat0), requires_grad=True)
        y_10 = y_hat.detach().clone()
        if K_sample <= 1:
            y_K0 = y_hat.detach().clone()

        for k in range(K_sample):
            y_hat_avg = torch.zeros_like(y_hat)
            y_step_sizek = y_step_size0 / (2 ** (k + 1))

            for j in range(2 ** k):
                z_Tr_j = next(iter(train_data_loader))[0]

                outputs = self.model(y_hat)
                loss_y = get_loss(outputs, z_Tr_j, self.pro_params)
                Loss = Lambda * torch.sum((y_hat - x_Tr) ** 2) - torch.mean(loss_y)
                grad_y = torch.autograd.grad(Loss, y_hat)[0]

                y_hat = y_hat - y_step_sizek * grad_y
                y_hat_avg = y_hat_avg * (1 - 1 / (j + 1)) + 1 / (j + 1) * y_hat.detach().clone()

            y_hat = Variable(to_tensor(y_hat_avg), requires_grad=True)

            if k == K_sample - 2:
                y_K0 = y_hat.detach().clone()
            if k == K_sample - 1:
                y_K10 = y_hat.detach().clone()

        y_10 = Variable(to_tensor(y_10), requires_grad=False)
        y_K0 = Variable(to_tensor(y_K0), requires_grad=False)
        y_K10 = Variable(to_tensor(y_K10), requires_grad=False)

        return y_10, y_K0, y_K10

    def get_eval_loss(self, model, X_train, Y_train):
        # Shift to evaluation mode
        model.eval()

        # Get evaluation size and dataset
        N1 = self.params.N1_EVAL

        indices_n1 = torch.randint(0, self.n_train, (N1,), device=self.device)
        X_eval = X_train[indices_n1]
        Y_eval = Y_train[indices_n1]

        # Get ERM loss for eval
        loss = 0
        z = model(X_eval)  # Get: [n_train, d_z]

        if self.pro_params['problem_name'] == 'Newsvendor':
            with torch.no_grad():
                loss = newsvendor_loss(z, Y_eval, self.pro_params['h_cost'], self.pro_params['b_cost'])

        elif self.pro_params['problem_name'] == 'Portfolio':
            with torch.no_grad():
                z_decision, beta = parse_mv_output(z, self.d_z)
                loss = portfolio_loss_mv(z_decision, Y_eval, beta, self.pro_params['eta'])

        elif self.pro_params['problem_name'] == 'Inventory':
            with torch.no_grad():
                loss = DifferentiableGurobiSolver.apply(z, Y_eval, self.pro_params)

         # Return mean loss as a tensor
        return loss.mean()

def to_tensor(x):
    if type(x) == np.ndarray:
        return torch.from_numpy(x).float()
    elif type(x) == torch.Tensor:
        return x
    else:
        print("Type error. Input should be either numpy array or torch tensor")

