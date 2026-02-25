"""
    This file includes SCSC for the Causal-SDRO.
"""

import torch.distributions as dist
from tqdm import tqdm
from CausalSDRO.Tools.Functions import *

'''
    Solving a Causal-SDRO problem by SCSC
'''

class CSDRO_Trainer:
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
        self.y_2 = torch.tensor([1.0], device=device, dtype=torch.float64)
        self.t2_theta_prev = torch.tensor([0.0], device=device, dtype=torch.float64)
        self.t3_theta_prev = torch.tensor([0.0], device=device, dtype=torch.float64)

        # For SRF, we use Adam as the optimizer
        self.optimizer = None
        if self.model_type == 'SRF' and self.params.use_adam == True:

            # Divide the params into leaf and non-leaf parts
            params_structure = []
            params_leaf = []

            for name, param in self.model.named_parameters():
                if 'internal' in name or 'routing_bn' in name:
                    params_structure.append(param)
                else:
                    params_leaf.append(param)

            # Using Adam, we set different learning rate for leaf and interval nodes
            # We also add l2-regularization to avoid overfitting
            structure_lr_scale = 10.0

            self.optimizer = torch.optim.Adam([
                {
                    'params': params_structure,
                    'lr': self.alpha_k * structure_lr_scale,
                    'weight_decay': 1e-4,
                    'name': 'structure'
                },
                {
                    'params': params_leaf,
                    'lr': self.alpha_k,
                    'weight_decay': 1e-3,
                    'name': 'leaf'
                }
            ])

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
        pbar_scsc = tqdm(range(self.epochs), desc=f"Training SCSC ({self.model_type})", leave=False)
        # Update parameters by SCSC
        for epoch in pbar_scsc:
            # Load batches
            for x_batch, y_batch in self.train_loader:

                self.train_step(x_batch, y_batch)

            if (epoch+100) % 400 == 0:
                self.alpha_k /= 2
                self.beta_k /= 2

            # Evaluate (every 5 epochs)
            if (epoch + 1) % 5 == 0:

                self.model.eval()

                with torch.no_grad():
                    # Input full dataset
                    eval_loss = self.get_eval_loss(self.model, X_train, Y_train)
                # --- Save loss history ---
                self.train_loss_history.append(eval_loss.item())
                pbar_scsc.set_postfix(TestDRO=eval_loss.item())

        return self.model, self.train_loss_history


    # Update parameters for each batch
    def train_step(self, X_train, Y_train):

        batch_size = X_train.shape[0]
        d_z = self.pro_params['dimension'][2]

        # Start training!
        self.model.train()

        # Step 0: Get hat_x^k and hat_y^k, then move to GPU
        X_train = X_train.to(device=self.device, dtype=torch.float64)
        Y_train = Y_train.to(device=self.device, dtype=torch.float64)

        ## Step 1: Get \xi_1^k and \xi_2^k
        # Sample \xi_1 and \xi_2 from kernel distribution Q (dimension: d_x) and W (dimension: d_y)
        xi_1 = self.sample_kernels(batch_size, self.d_x)  # [batch_size, d_x]
        xi_2 = self.sample_kernels(batch_size, self.d_y)  # [batch_size, d_y]

        ## Step 2: Get t3(theta), and t_2(t_3(theta)), calculate by the average of all samples
        x_perturbed = X_train + xi_1   # [batch_size, d_x]
        y_perturbed = Y_train + xi_2   # [batch_size, d_y]
        # decision
        z = self.model(x_perturbed).reshape(-1, d_z) # [batch_size, d_z]
        # Get loss value
        loss_for_t3 = get_loss(z, y_perturbed, self.pro_params)
        el = max(self.lam * self.epsilon, 1e-6)
        STABLE_LOSS_MAX = self.params.scalar * el
        # Loss value!
        stable_loss = torch.clamp(loss_for_t3, max=STABLE_LOSS_MAX)

        # Get mean t3(theta), just like the idea of mini-batch SGD
        with torch.no_grad():
            t3_theta_k_stable = torch.exp(stable_loss.detach() / el).mean()
        # Get t2(t2(theta)), here n_{hat_x} = 1, and thus
        t2_theta_k_stable = t3_theta_k_stable

        ## Step 3:  Update y_2 and y_1
        # Set a limitation for y, avoid gradient explosion
        Y_MAX_LIMIT = 1e50

        # update y_2
        y2_update_term = self.y_2 + t3_theta_k_stable - self.t3_theta_prev
        if torch.isnan(y2_update_term) or torch.isinf(y2_update_term):
            # Avoid gradient explosion
            self.y_2 = (1 - self.beta_k) * self.y_2 + self.beta_k * t3_theta_k_stable
        else:
            self.y_2 = (1 - self.beta_k) * y2_update_term + self.beta_k * t3_theta_k_stable
        self.y_2 = torch.clamp(self.y_2, min=1e-8, max=Y_MAX_LIMIT)  # Limitation

        # update y_1
        y1_update_term = self.y_1 + t2_theta_k_stable - self.t2_theta_prev
        if torch.isnan(y1_update_term) or torch.isinf(y1_update_term):
            self.y_1 = (1 - self.beta_k) * self.y_1 + self.beta_k * t2_theta_k_stable
        else:
            self.y_1 = (1 - self.beta_k) * y1_update_term + self.beta_k * t2_theta_k_stable
        self.y_1 = torch.clamp(self.y_1, min=1e-8, max=Y_MAX_LIMIT)

        # Set historical k for k+1
        self.t3_theta_prev = t3_theta_k_stable
        self.t2_theta_prev = t2_theta_k_stable

        ## Step 4: Update gradients for t2 and t3
        # here, (\partial t1 (y1) / \partial y1) = 1 / y_1, (\partial t2 (y2) / \partial y2) = 1
        # (\partial t3 (psi/el) / \partial (psi) ) = (1/el) * exp(psi/el) = (1/el) * t3_theta_k_stable
        # (\partial psi (theta) / \partial theta) -> to be calculate
        # In F(theta), their is also a coefficient (el)
        gradient_coefficient = (t3_theta_k_stable.detach() / self.y_1.detach())
        # This mean loss is used for backpropagation
        if el < 1:   # This is the correct loss
            surrogate_loss = (stable_loss * gradient_coefficient.detach()).mean()
        else: # If el is very large, we control the value of el, i.e., we optimize an objective without constant el
            surrogate_loss = (stable_loss * gradient_coefficient.detach() / el).mean()
        # Update the gradient of all parameters by
        surrogate_loss.backward()

        # A clipper for gradient
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        ## Step 5: Update theta

        if self.model_type == 'SRF' and self.params.use_adam == True:
            self.optimizer.step()
        else:
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


    # Sample from kernel distribution Q or W, return a sample [N, dimension]
    def sample_kernels(self, N, dimension):
        # Sample from Laplace(0, epsilon) for p=1
        if self.norm == 1:
            # p=1 corresponds to the Laplace distribution
            loc = 0.0
            scale = self.epsilon
            laplace_dist = dist.Laplace(loc=loc, scale=scale)
            # .sample((N, d)) will correctly sample N vectors of dimension d
            samples = laplace_dist.sample((N, dimension)).to(self.device).to(dtype=torch.float64)
        # Sample from N(0, (epsilon/2) * I) for p=2
        elif self.norm == 2:
            # p=2 corresponds to the Gaussian distribution
            variance = self.epsilon / self.norm
            variance_tensor = torch.tensor(variance, device=self.device, dtype=torch.float64)

            mvn = dist.MultivariateNormal(torch.zeros(dimension, device=self.device, dtype=torch.float64),
                                          covariance_matrix=torch.eye(dimension, device=self.device,
                                                                      dtype=torch.float64) * variance_tensor)
            samples = mvn.sample((N,))
        else:
            raise NotImplementedError(f"Only p=1 and p=2 are implemented. Got p={self.norm}.")

        return samples

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