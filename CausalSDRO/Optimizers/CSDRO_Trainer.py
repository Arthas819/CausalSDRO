"""
    This file includes CM-SCSC for Causal-SDRO.
"""

import copy
import torch.distributions as dist
from tqdm import tqdm
from Tools.Functions import *


class CSDRO_Trainer:
    def __init__(self, model, train_loader, params, device, model_type):
        # Basic settings
        self.model = model.to(device).double()
        self.model_type = model_type
        self.train_loader = train_loader
        self.epochs = params.EPOCHS_SCSC
        self.params = params
        self.device = device
        self.alpha_k = params.LR_alpha
        self.beta_k = params.LR_beta
        self.grad_clip = params.grad_clip
        self.track_step_condition = getattr(params, 'track_step_condition', False)
        self.train_loss_history = []

        # Store y_1, y_2 and y_2_prev in log scale, avoiding exp(B / (lambda * epsilon)).
        self.log_y_1 = None
        self.log_y_2 = None
        self.log_y_2_prev = None
        self.model_prev = None

        # Retain the original optional Adam settings for numerical experiments.
        self.optimizer = None
        if self.model_type == 'SRF' and self.params.use_adam == True:
            params_structure = []
            params_leaf = []
            for name, param in self.model.named_parameters():
                if 'internal' in name or 'routing_bn' in name:
                    params_structure.append(param)
                else:
                    params_leaf.append(param)
            self.optimizer = torch.optim.Adam([
                {'params': params_structure, 'lr': self.alpha_k * getattr(self.params, 'structure_lr_scale', 10.0),
                 'weight_decay': 0.1 * getattr(self.params, 'adam_decay', 1e-3), 'name': 'structure'},
                {'params': params_leaf, 'lr': self.alpha_k,
                 'weight_decay': getattr(self.params, 'adam_decay', 1e-3), 'name': 'leaf'}
            ])

    def train_scsc(self, X_train, Y_train, pro_params, progress=True,
                   checkpoint_callback=None, checkpoint_every=100, evaluate_every=5):
        if pro_params['problem_name'] == 'Inventory':
            self.model.configure_inventory(pro_params)
        X_train = X_train.to(device=self.device, dtype=torch.float64)
        Y_train = Y_train.to(device=self.device, dtype=torch.float64)
        self.n_train, self.d_x = X_train.shape
        self.d_y = Y_train.shape[1]
        self.d_z = pro_params['dimension'][2]
        self.pro_params = pro_params
        self.norm = pro_params['norm']
        self.epsilon = pro_params['epsilon']
        self.lam = pro_params['lambda']
        self.el = self.lam * self.epsilon

        # Bounded training losses use the projection interval from Algorithm 1.
        default_B = {'Newsvendor': 10.0, 'Inventory': 50.0, 'Portfolio': 100.0}.get(pro_params['problem_name'])
        self.B = pro_params.get('B', default_B)
        if self.B is None:
            self.log_y_1_min, self.log_y_1_max = math.log(1e-8), math.log(1e50)
        else:
            self.log_y_1_min, self.log_y_1_max = 0.0, self.B / self.el

        # Keep the original epoch interface: K updates in total, Q inner samples per update.
        self.K = self.epochs * len(self.train_loader)
        self.Q = int(getattr(self.params, 'Q', math.ceil(self.K ** 0.25)))

        # Use supplied Portfolio groups or group identical covariates; p_m = n_m / N.
        if pro_params.get('causal_groups', True):
            if pro_params['problem_name'] == 'Portfolio' and 'group_index' in pro_params:
                group_index = pro_params['group_index'].to(self.device)
                counts = torch.bincount(group_index)
                centers = torch.stack([torch.where(group_index == m)[0][0] for m in range(len(counts))])
                self.X_groups = X_train[centers]
            else:
                self.X_groups, group_index, counts = torch.unique(
                    X_train, dim=0, return_inverse=True, return_counts=True)
        else:
            # Compatibility with the SDRO baseline: one group per joint observation.
            self.X_groups = X_train
            group_index = torch.arange(self.n_train, device=self.device)
            counts = torch.ones(self.n_train, device=self.device, dtype=torch.long)
        self.M = len(self.X_groups)
        self.group_index = group_index
        self.group_counts = counts
        self.p_m = counts.to(dtype=torch.float64) / self.n_train
        self.group_indices = [torch.where(group_index == m)[0] for m in range(self.M)]
        self.Y_train = Y_train
        if pro_params['problem_name'] == 'Portfolio' and getattr(self.params, 'outer_batch', 1) > 1:
            self.group_members = torch.stack(self.group_indices)

        # Algorithm 1, Step 1: theta_prev = theta_0, y_1,m = 1.
        self.model_prev = copy.deepcopy(self.model)
        self.model_prev.requires_grad_(False)
        self.log_y_1 = torch.zeros(self.M, device=self.device, dtype=torch.float64)
        # Two N-dimensional buffers; only the current group's entries enter t_2,m.
        self.log_y_2 = torch.zeros(self.n_train, device=self.device, dtype=torch.float64)
        self.log_y_2_prev = torch.zeros_like(self.log_y_2)
        self.train_loss_history = []
        self.sampled_Cd_max = torch.zeros((), device=self.device, dtype=torch.float64)

        # Initialize the tracker at the fitted warm-start weights when requested.
        # model_prev is already an exact copy of these weights.
        tracking_samples = int(getattr(self.params, 'warm_start_tracking_samples', 0))
        if tracking_samples > 0 and self.beta_k != 1:
            with torch.no_grad():
                log_sum = torch.full_like(self.log_y_1, -torch.inf)
                for _ in range(tracking_samples):
                    xi_1 = self.sample_kernels(1, self.d_x)
                    xi_2 = (torch.rand(self.Q, self.d_y, device=self.device, dtype=torch.float64)
                            if pro_params['problem_name'] == 'Inventory'
                            else self.sample_kernels(self.Q, self.d_y))
                    _, log_H = self.get_log_H_all(self.model, xi_1, xi_2)
                    log_sum = torch.logaddexp(log_sum, log_H)
                self.log_y_1.copy_((log_sum - math.log(tracking_samples)).clamp(
                    self.log_y_1_min, self.log_y_1_max))

        step = 0
        self.completed_steps = 0
        pbar_scsc = tqdm(range(self.epochs), desc=f"Training CM-SCSC ({self.model_type})",
                        leave=False, disable=not progress)
        for epoch in pbar_scsc:
            # Each update samples a group from the full empirical distribution.
            for _ in range(len(self.train_loader)):
                self.train_step()
                step += 1
                self.completed_steps = step
                if checkpoint_callback is not None and (step % checkpoint_every == 0 or step in (1, 10, 25, 50)):
                    if checkpoint_callback(self.model, step) is False:
                        return self.model, self.train_loss_history

            # Keep the supplied alpha and beta fixed, as in Theorem 7.
            # Set evaluate_every=0 for timing runs without diagnostic evaluation.
            if evaluate_every and (epoch + 1) % evaluate_every == 0:
                self.model.eval()
                with torch.no_grad():
                    eval_loss = self.get_eval_loss(self.model, X_train, Y_train)
                self.train_loss_history.append(eval_loss.item())
                pbar_scsc.set_postfix(EvalLoss=eval_loss.item())

        if checkpoint_callback is not None and step % checkpoint_every:
            checkpoint_callback(self.model, step)
        return self.model, self.train_loss_history

    def get_log_y_2(self, model, m, xi_1, xi_2):
        # Steps 5-6: y_2,m,j = (1/Q) * sum_q exp(loss_jq / (lambda * epsilon)).
        indices = self.group_indices[m]
        n_m = len(indices)
        x = self.X_groups[m:m+1] + xi_1
        y = self.perturb_y(self.Y_train[indices], xi_2).reshape(-1, self.d_y)
        z = model(x).reshape(1, self.d_z).expand(n_m * self.Q, -1)
        loss = get_loss(z, y, self.pro_params).reshape(n_m, self.Q)
        if self.B is None:
            # Preserve the original Inventory loss cap.
            loss = torch.clamp(loss, max=self.params.scalar * self.el)
        return torch.logsumexp(loss / self.el, dim=1) - math.log(self.Q)

    def get_log_H_all(self, model, xi_1, xi_2):
        # Evaluate all groups in one batch, using the same perturbations as Algorithm 1.
        z = model(self.X_groups + xi_1).reshape(self.M, self.d_z)[self.group_index]
        z = z[:, None, :].expand(-1, self.Q, -1).reshape(-1, self.d_z)
        y = self.perturb_y(self.Y_train, xi_2).reshape(-1, self.d_y)
        loss = get_loss(z, y, self.pro_params).reshape(self.n_train, self.Q)
        if self.B is None:
            loss = torch.clamp(loss, max=self.params.scalar * self.el)
        log_y_2 = torch.logsumexp(loss / self.el, dim=1) - math.log(self.Q)
        log_H = torch.zeros_like(self.log_y_1).scatter_add_(0, self.group_index, log_y_2)
        return log_y_2, log_H / self.group_counts

    def train_step(self):
        if (self.pro_params['problem_name'] == 'Portfolio' and self.model_type == 'SRF'
                and getattr(self.params, 'outer_batch', 1) > 1):
            return self.train_step_batch()
        self.model.train()
        self.model_prev.train()

        # Step 3: m_k, xi_1 and Q i.i.d. xi_2 samples are mutually independent.
        m_k = torch.multinomial(self.p_m, 1).item()
        xi_1 = self.sample_kernels(1, self.d_x)
        xi_2 = (torch.rand(self.Q, self.d_y, device=self.device, dtype=torch.float64)
                if self.pro_params['problem_name'] == 'Inventory'
                else self.sample_kernels(self.Q, self.d_y))

        self.model.zero_grad()
        if self.beta_k == 1:
            # Here y_1 = H; only the sampled group enters this or future updates.
            log_H_k = self.get_log_y_2(self.model, m_k, xi_1, xi_2).mean()
            with torch.no_grad():
                self.log_y_1[m_k] = log_H_k.detach().clamp(self.log_y_1_min, self.log_y_1_max)
        else:
            # Steps 4-9: update every group with the same perturbations.
            with torch.no_grad():
                self.log_y_2, log_H_k = self.get_log_H_all(self.model, xi_1, xi_2)
                self.log_y_2_prev, log_H_prev = self.get_log_H_all(self.model_prev, xi_1, xi_2)
                log_positive = torch.logaddexp(
                    self.log_y_1 + math.log1p(-self.beta_k), log_H_k)
                log_negative = log_H_prev + math.log1p(-self.beta_k)
                difference = torch.clamp(log_negative - log_positive, max=0)
                log_update = log_positive + torch.log(-torch.expm1(difference))
                self.log_y_1 = torch.clamp(log_update, min=self.log_y_1_min, max=self.log_y_1_max)
            log_H_k = self.get_log_y_2(self.model, m_k, xi_1, xi_2).mean()

        # Step 10: (lambda * epsilon) * grad H_mk / y_1,mk.
        (self.el * log_H_k).backward()
        # grad H = H * grad(log H); H/y_1 is a detached chain-rule coefficient.
        with torch.no_grad():
            gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
            log_scale = log_H_k.detach() - self.log_y_1[m_k]
            if self.track_step_condition:
                # Sampled lambda*epsilon*||grad H||, before division by y_1 or clipping.
                # This is an empirical diagnostic, not the uniform C_d in the proof.
                base_norm = torch.stack([g.norm() for g in gradients]).norm()
                sampled_Cd = torch.exp(torch.log(base_norm) + log_H_k.detach())
                self.sampled_Cd_max = torch.maximum(self.sampled_Cd_max, sampled_Cd)
            if self.grad_clip is not None:
                grad_norm = torch.stack([g.norm() for g in gradients]).norm()
                log_scale = torch.minimum(log_scale,
                    torch.log(grad_norm.new_tensor(self.grad_clip)) - torch.log(grad_norm))
            for gradient in gradients:
                # Evaluate the product in log scale, preserving zero gradients.
                gradient.copy_(gradient.sign() * torch.exp(torch.log(gradient.abs()) + log_scale))

        # Store theta^k before applying the update, for the next same-sample correction.
        if self.beta_k != 1:
            self.model_prev.load_state_dict(self.model.state_dict())
        if self.optimizer is not None:
            self.optimizer.step()
        else:
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.grad is not None:
                        param -= self.alpha_k * param.grad

    # Average independent group gradients when beta=1.
    def batch_log_H(self, groups, xi_1, xi_2):
        if not hasattr(self, 'group_members'):
            self.group_members = torch.stack(self.group_indices)
        indices = self.group_members[groups]
        batch_size, n_m = indices.shape
        z = self.model(self.X_groups[groups] + xi_1)
        z = z[:, None, None, :].expand(-1, n_m, self.Q, -1).reshape(-1, self.d_z)
        y = (self.Y_train[indices, None, :] + xi_2[:, None, :, :]).reshape(-1, self.d_y)
        loss = get_loss(z, y, self.pro_params).reshape(batch_size, n_m, self.Q)
        return (torch.logsumexp(loss / self.el, dim=2) - math.log(self.Q)).mean(dim=1)

    def train_step_batch(self):
        batch_size = int(getattr(self.params, 'outer_batch', 1))
        if self.pro_params['problem_name'] != 'Portfolio' or self.beta_k != 1:
            raise ValueError('Portfolio outer batches require beta=1.')
        self.model.train()
        groups = torch.multinomial(self.p_m, batch_size, replacement=True)
        xi_1 = self.sample_kernels(batch_size, self.d_x)
        xi_2 = self.sample_kernels(batch_size * self.Q, self.d_y).reshape(batch_size, self.Q, self.d_y)
        self.model.zero_grad()
        log_H = self.batch_log_H(groups, xi_1, xi_2)
        (self.el * log_H.mean()).backward()
        with torch.no_grad():
            self.log_y_1[groups] = log_H.detach().clamp(self.log_y_1_min, self.log_y_1_max)
            if self.grad_clip is not None:
                gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
                norm = torch.stack([g.norm() for g in gradients]).norm()
                scale = (self.grad_clip / norm).clamp(max=1)
                for gradient in gradients:
                    gradient.mul_(scale)
        if self.optimizer is not None:
            self.optimizer.step()
        else:
            with torch.no_grad():
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        parameter -= self.alpha_k * parameter.grad

    def perturb_y(self, y, xi_2):
        if self.pro_params['problem_name'] == 'Inventory':
            return inventory_kernel_y(y[:, None, :], xi_2[None, :, :], self.epsilon, self.norm)
        return y[:, None, :] + xi_2[None, :, :]

    def sample_kernels(self, N, dimension):
        # p=1: Laplace(0, epsilon); p=2: Normal(0, (epsilon/2) * I).
        if self.norm == 1:
            loc = torch.tensor(0.0, device=self.device, dtype=torch.float64)
            scale = torch.tensor(self.epsilon, device=self.device, dtype=torch.float64)
            return dist.Laplace(loc, scale).sample((N, dimension))
        elif self.norm == 2:
            return math.sqrt(self.epsilon / 2) * torch.randn(
                N, dimension, device=self.device, dtype=torch.float64)
        else:
            raise NotImplementedError(f"Only p=1 and p=2 are implemented. Got p={self.norm}.")

    def get_eval_loss(self, model, X_train, Y_train):
        # Preserve the original evaluation: average loss on sampled historical observations.
        model.eval()
        indices = torch.randint(0, self.n_train, (self.params.N1_EVAL,), device=self.device)
        with torch.no_grad():
            z = model(X_train[indices]).reshape(-1, self.d_z)
            return get_loss(z, Y_train[indices], self.pro_params).mean()
