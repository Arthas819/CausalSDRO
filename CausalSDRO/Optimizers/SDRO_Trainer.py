"""
    This file Solves SDRO model by SCSC.
"""

import torch.distributions as dist
from tqdm import tqdm
from CausalSDRO.Tools.Functions import *

class SDRO_Trainer:
    def __init__(self, model, train_loader, params, device, model_type):
        # Basic settings
        self.model =  model.to(device)
        self.model_type = model_type
        self.train_loader = train_loader
        self.epochs = params.EPOCHS_SCSC
        self.params = params
        self.device = device
        self.alpha_k = params.LR_alpha
        self.beta_k = params.LR_beta
        self.grad_clip = params.grad_clip

        # A list to store loss history
        self.train_loss_history = []

        # SCSC state variables
        self.y_1 = torch.tensor([1.0], device=device, dtype=torch.float64)
        self.t2_theta_prev = torch.tensor([0.0], device=device, dtype=torch.float64)


    def train_scsc(self, X_train, Y_train, pro_params):

        self.n_train = X_train.shape[0]
        self.d_x = X_train.shape[1]
        self.d_y = Y_train.shape[1]
        self.d_z = pro_params['dimension'][2]

        self.pro_params = pro_params
        self.norm = pro_params['norm']
        self.epsilon = pro_params['epsilon']
        self.lam = pro_params['lambda']

        # Set a bar
        pbar_scsc = tqdm(range(self.epochs), desc=f"Training SCSC for SDRO ({self.model_type})", leave=False)
        # Update parameters by SCSC
        for epoch in pbar_scsc:
            # Load batches
            for x_batch, y_batch in self.train_loader:
                self.train_step(x_batch, y_batch)

            if (epoch + 100) % 400 == 0:
                self.alpha_k /= 2
                self.beta_k /= 2

            # Evaluate (every 5 epochs)
            if (epoch + 1) % 5 == 0:
                self.model.eval()

                with torch.no_grad():
                    # Input full dataset
                    eval_loss = self.get_eval_loss(self.model, X_train, Y_train)
                # --- Save loss history ---
                # loss.item() -> change to a scalar
                self.train_loss_history.append(eval_loss.item())
                pbar_scsc.set_postfix(TestDRO=eval_loss.item())

        return self.model, self.train_loss_history

    def train_step(self, X_train, Y_train):

        batch_size = X_train.shape[0]
        d_x = X_train.shape[1]
        d_y = Y_train.shape[1]
        d_z = self.pro_params['dimension'][2]

        # Start training!
        self.model.train()

        # Get training data
        X_train = X_train.to(device=self.device, dtype=torch.float64)
        Y_train = Y_train.to(device=self.device, dtype=torch.float64)

        xi_3, xi_4 = self.sample_kernels_xi3and4(batch_size, d_x, d_y)

        x_perturbed = X_train + xi_3   # [batch_size, d_x]
        y_perturbed = Y_train + xi_4   # [batch_size, d_y]
        # decision
        # self.model.zero_grad()
        z = self.model(x_perturbed).reshape(-1, d_z) # [batch_size, d_z]
        # Get loss value
        loss_value = get_loss(z, y_perturbed, self.pro_params)
        el = max(self.lam * self.epsilon, 1e-6)
        STABLE_LOSS_MAX = 80.0 * el
        # Loss value!
        stable_loss = torch.clamp(loss_value, max=STABLE_LOSS_MAX)

        # Calculate inner expectation loss, t2
        with torch.no_grad():
            #### Attention
            t2_theta_k = torch.exp(stable_loss.detach() / el)

        t2_theta_k_stable = t2_theta_k.mean()

        # Update y1
        # Set a limitation for y
        Y_MAX_LIMIT = 1e50
        y1_update_term = self.y_1 + t2_theta_k_stable - self.t2_theta_prev
        if torch.isnan(y1_update_term) or torch.isinf(y1_update_term):
            self.y_1 = (1 - self.beta_k) * self.y_1 + self.beta_k * t2_theta_k_stable
        else:
            self.y_1 = (1 - self.beta_k) * y1_update_term + self.beta_k * t2_theta_k_stable
        self.y_1 = torch.clamp(self.y_1, min=1e-8, max=Y_MAX_LIMIT)
        # Update t2
        self.t2_theta_prev = t2_theta_k_stable
        # Update the gradient of t2
        # here, (\partial t1 (y1) / \partial y1) = 1 / y_1,
        # (\partial t2 (psi/el) / \partial (psi) ) = (1/el) * exp(psi/el) = (1/el) * t2_theta_k_stable
        # (\partial psi (theta) / \partial theta) -> to be calculate
        gradient_coefficient = (t2_theta_k_stable.detach() / self.y_1.detach())
        # Mean loss used for backpropagation
        if el < 1:   # This is the correct loss
            surrogate_loss = (stable_loss * gradient_coefficient.detach()).mean()
        else: # If el is very large, we control the value of el, i.e., we optimize an objective without constant el
            surrogate_loss = (stable_loss * gradient_coefficient.detach() / el).mean()
        # Update the gradient of all parameters by
        surrogate_loss.backward()

        # A clipper for gradient
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"Warning: NaN gradient detected in SCSC (L_max={STABLE_LOSS_MAX:.2f}). Skipping update.")
                        self.model.zero_grad()
                        break

                    # l2_grad = self.l2_reg_strength * param.data
                    # total_grad = param.grad + l2_grad
                    total_grad = param.grad
                    param -= self.alpha_k * total_grad

    def sample_kernels_xi3and4(self, N, d_x, d_y):
        # Sample from Laplace(0, epsilon) for p=1
        if self.norm == 1:
            loc = 0.0
            scale = self.epsilon
            laplace_dist = dist.Laplace(loc=loc, scale=scale)
            xi_3 = laplace_dist.sample((N, d_x)).to(self.device).to(dtype=torch.float64)
            xi_4 = laplace_dist.sample((N, d_y)).to(self.device).to(dtype=torch.float64)
        # Sample from N(0, (epsilon/2) * I) for p=2
        elif self.norm == 2:
            variance = self.epsilon / 2.0
            variance_tensor = torch.tensor(variance, device=self.device, dtype=torch.float64)
            mvn_x = dist.MultivariateNormal(
                torch.zeros(d_x, device=self.device, dtype=torch.float64),
                covariance_matrix=torch.eye(d_x, device=self.device, dtype=torch.float64) * variance_tensor
            )
            xi_3 = mvn_x.sample((N,))
            mvn_y = dist.MultivariateNormal(
                torch.zeros(d_y, device=self.device, dtype=torch.float64),
                covariance_matrix=torch.eye(d_y, device=self.device, dtype=torch.float64) * variance_tensor
            )
            xi_4 = mvn_y.sample((N,))
        else:
            raise NotImplementedError(f"Only p=1 and p=2 are implemented. Got p={self.norm}.")
        return xi_3, xi_4

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