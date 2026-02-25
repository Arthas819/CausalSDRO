"""
    This file Solves KL-DRO model by standard SGD.
"""

from tqdm import tqdm
from CausalSDRO.Tools.Functions import *

class KLDRO_Trainer:
    def __init__(self, model, train_loader, params, device, model_type):
        # Basic settings
        self.model =  model.to(device).double()
        self.model_type = model_type
        self.train_loader = train_loader
        self.params = params
        self.device = device
        self.epochs = params.EPOCHS_SCSC

        # A list to store loss history
        self.train_loss_history = []

        # Optimizer
        lr = 0.01
        weight_decay = 1e-4
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    def train_sgd(self, X_train, Y_train, pro_params):

        self.n_train = X_train.shape[0]
        self.d_x = X_train.shape[1]
        self.d_y = Y_train.shape[1]
        self.d_z = pro_params['dimension'][2]

        self.pro_params = pro_params
        self.lam = pro_params['lambda']
        # self.norm = pro_params['norm']
        # self.epsilon = pro_params['epsilon']

        # Set a bar
        pbar_sgd = tqdm(range(self.epochs), desc=f"Training SGD for KL-DRO ({self.model_type})", leave=False)

        # Update parameters by SGD
        for epoch in pbar_sgd:

            # Load batches
            for x_batch, y_batch in self.train_loader:

                self.train_step(x_batch, y_batch)

            # Evaluate (every 5 epochs)
            if (epoch + 1) % 5 == 0:

                self.model.eval()

                with torch.no_grad():
                    # Input full dataset
                    eval_loss = self.get_eval_loss(self.model, X_train, Y_train)

                # Record historical loss
                self.train_loss_history.append(eval_loss.item())
                pbar_sgd.set_postfix(TestDRO=eval_loss.item())

        return self.model, self.train_loss_history

    # Update parameters for each batch
    def train_step(self, X_train, Y_train):

        batch_size = X_train.shape[0]
        d_x = X_train.shape[1]
        d_y = Y_train.shape[1]
        d_z = self.pro_params['dimension'][2]

        # Start training!
        self.model.train()

        # Get hat_x^k and hat_y^k, then move to GPU
        X_train = X_train.to(device=self.device, dtype=torch.float64)
        Y_train = Y_train.to(device=self.device, dtype=torch.float64)

        # Get a decision
        z = self.model(X_train).reshape(-1, d_z) # [batch_size, d_z]
        # Get news loss
        loss_value = get_loss(z, Y_train, self.pro_params)
        scaled_loss = loss_value / self.lam
        # log-(1/N)-sum-exp(psi / lam)  ->  log-sum-exp(psi) - log(1/N)
        total_loss = self.lam * (torch.logsumexp(scaled_loss, dim=0) - np.log(batch_size))
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

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
