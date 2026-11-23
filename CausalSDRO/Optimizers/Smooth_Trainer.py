"""
    Training the kernel-smoothing baseline in our Appendix G.4.
"""

import copy
import torch.distributions as dist
from Optimizers.ERM_Trainer import *
from Optimizers.CSDRO_Trainer import *


class Smooth_Trainer(ERM_Trainer):

    def train_smooth(self, X_train, Y_train, pro_params, progress=True,
                     checkpoint_callback=None, checkpoint_every=100):
        self.norm = pro_params['norm']
        self.epsilon = pro_params['epsilon']
        if pro_params['problem_name'] == 'Inventory':
            self.pro_params = pro_params
            self.model.configure_inventory(pro_params)
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.params.LR_SMOOTH)
            for step in tqdm(range(self.params.K_SMOOTH), desc='Training Smooth (SGD)',
                             leave=False, disable=not progress):
                indices = torch.randint(len(X_train), (self.params.BATCH_SIZE_SMOOTH,),
                                        device=X_train.device)
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.get_erm_loss(self.model, X_train[indices], Y_train[indices])
                loss.backward()
                self.optimizer.step()
                if checkpoint_callback is not None and (step + 1) % checkpoint_every == 0:
                    if checkpoint_callback(self.model, step + 1) is False:
                        return self.model, self.train_loss_history
                if (step + 1) % 100 == 0:
                    self.train_loss_history.append(loss.item())
            if checkpoint_callback is not None and self.params.K_SMOOTH % checkpoint_every:
                checkpoint_callback(self.model, self.params.K_SMOOTH)
            return self.model, self.train_loss_history

        # Preserve the existing Newsvendor training configuration.
        return self.train_erm(X_train, Y_train, pro_params, progress=progress,
                              checkpoint_callback=checkpoint_callback,
                              checkpoint_every=checkpoint_every)

    def get_erm_loss(self, model, X_train, Y_train):
        if self.epsilon == 0:
            return super().get_erm_loss(model, X_train, Y_train)

        if self.norm == 1:
            kernel = dist.Laplace(X_train.new_tensor(0.0), X_train.new_tensor(self.epsilon))
            xi_1 = kernel.sample(X_train.shape)
        else:
            sigma = math.sqrt(self.epsilon / 2)
            xi_1 = sigma * torch.randn_like(X_train)

        if self.pro_params['problem_name'] == 'Inventory':
            Y_smooth = inventory_kernel_y(Y_train, torch.rand_like(Y_train), self.epsilon, self.norm)
        else:
            xi_2 = kernel.sample(Y_train.shape) if self.norm == 1 else sigma * torch.randn_like(Y_train)
            Y_smooth = Y_train + xi_2
        # Average perturbed losses directly; no exponential or lambda.
        return super().get_erm_loss(model, X_train + xi_1, Y_smooth)


def validation_data(data_generator, size=1000):
    """Same population and training normalization, with a separate seed."""
    previous_generator = data_generator.generator
    data_generator.generator = torch.Generator(device=data_generator.device).manual_seed(
        data_generator.seed + 200000)
    try:
        X, Y = data_generator.generate(size)
    finally:
        data_generator.generator = previous_generator
    X = (X - data_generator.X_train_mean) / data_generator.X_train_std
    return X, Y


def evaluate_loss(model, X, Y, pro_params, batch_size=500):
    return math.fsum(get_test_loss(model, X[i:i + batch_size], Y[i:i + batch_size],
                                  pro_params) * len(X[i:i + batch_size])
                     for i in range(0, len(X), batch_size)) / len(X)


def fit_smooth_csdro(model, train_loader, params, device, X_train, Y_train,
                     X_validation, Y_validation, pro_params, progress=True):

    seed = int(pro_params.get('seed', 42))
    every = int(getattr(params, 'WARM_CHECKPOINT_EVERY', 100))
    lr_factors = tuple(getattr(params, 'WARM_LR_FACTORS', (0.03, 0.1, 0.3)))

    smooth = copy.deepcopy(model)
    if pro_params['problem_name'] == 'Inventory':
        smooth.configure_inventory(pro_params)
    best_value = evaluate_loss(smooth, X_validation, Y_validation, pro_params)
    if not math.isfinite(best_value):
        raise ValueError('Nonfinite initial validation loss.')
    best_state = copy.deepcopy(smooth.state_dict())
    best_step = 0

    def select_smooth(candidate, step):
        nonlocal best_value, best_state, best_step
        value = evaluate_loss(candidate, X_validation, Y_validation, pro_params)
        if not math.isfinite(value):
            return False
        if value < best_value:
            best_value, best_state, best_step = value, copy.deepcopy(candidate.state_dict()), step

    # Restore the caller's CPU/CUDA random stream after the two training stages.
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        trainer = Smooth_Trainer(smooth, train_loader, copy.deepcopy(params), device, 'SRF')
        trainer.train_smooth(X_train, Y_train, pro_params, progress=progress,
                             checkpoint_callback=select_smooth, checkpoint_every=every)
        smooth.load_state_dict(best_state)
        smooth_value, smooth_step = best_value, best_step
        refined_state, refined_step, refined_factor = copy.deepcopy(best_state), 0, 0.0

        for factor in lr_factors:
            trial_params = copy.deepcopy(params)
            trial_params.LR_alpha *= factor
            trial_params.warm_start_tracking_samples = 16
            # Every trial starts from the exact same fitted smooth weights.
            torch.manual_seed(seed + 10000)
            trainer = CSDRO_Trainer(copy.deepcopy(smooth), train_loader,
                                    trial_params, device, 'SRF')

            def select_csdro(candidate, step):
                nonlocal best_value, refined_state, refined_step, refined_factor
                value = evaluate_loss(candidate, X_validation, Y_validation, pro_params)
                if not math.isfinite(value):
                    return False
                if value < best_value:
                    best_value = value
                    refined_state = copy.deepcopy(candidate.state_dict())
                    refined_step, refined_factor = step, factor

            trainer.train_scsc(X_train, Y_train, pro_params, progress=progress,
                                    checkpoint_callback=select_csdro, checkpoint_every=every)

    refined = copy.deepcopy(smooth)
    refined.load_state_dict(refined_state)
    norm = int(pro_params['norm'])
    diagnostics = {
        f'Validation_{norm}_Smooth': smooth_value,
        f'Validation_{norm}_CSDRO': best_value,
        f'Smooth_{norm}_Selected_Step': smooth_step,
        f'CSDRO_{norm}_Selected_Step': refined_step,
        f'CSDRO_{norm}_LR_Factor': refined_factor,
    }
    return smooth, refined, diagnostics
