"""
    This file includes SAA for solving ERM model.
"""

import torch.optim as optim
from tqdm import tqdm
from Tools.Functions import *


'''
    SAA for solving an ERM problem, serve as a baseline
'''
class ERM_Trainer:

    def __init__(self, model, train_loader, params, device, model_type):

        # Basic settings
        self.model =  model.to(device)
        self.model_type = model_type
        self.train_loader = train_loader
        self.epochs = params.EPOCHS_SAA
        self.params = params
        self.device = device

        # Use Adam optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=params.LR_SAA)
        # A list to store loss history
        self.train_loss_history = []

    # Train ERM model
    def train_erm(self, X_train, Y_train, pro_params, progress=True,
                  checkpoint_callback=None, checkpoint_every=100):
        if pro_params['problem_name'] == 'Inventory':
            self.model.configure_inventory(pro_params)
        self.n_train = X_train.shape[0]
        self.pro_params = pro_params
        self.d_z = pro_params['dimension'][2]

        # Set a bar
        step = 0
        pbar_saa = tqdm(range(self.epochs), desc=f"Training ERM ({self.model_type})",
                       leave=False, disable=not progress)
        # Update parameters by SAA
        for epoch in pbar_saa:
            # Load batches
            for x_batch, y_batch in self.train_loader:

                self.model.train()
                self.optimizer.zero_grad()

                loss = self.get_erm_loss(self.model, x_batch, y_batch)

                # Backpropagation
                loss.backward()

                # update parameters
                self.optimizer.step()
                step += 1
                if checkpoint_callback is not None and step % checkpoint_every == 0:
                    if checkpoint_callback(self.model, step) is False:
                        return self.model, self.train_loss_history

                # update the tqdm
                if (epoch + 1) % 20 == 0:
                    pbar_saa.set_postfix(Loss=loss.item())

            # Evaluate (every 5 epochs)
            if (epoch + 1) % 5 == 0:
                self.model.eval()
                with torch.no_grad():
                    # Input full dataset
                    eval_loss = self.get_eval_loss(self.model, X_train, Y_train)
                # --- Save loss history ---
                # loss.item() -> change to a scalar
                    self.train_loss_history.append(eval_loss.item())
                    pbar_saa.set_postfix(TestDRO=eval_loss.item())

        if checkpoint_callback is not None and step % checkpoint_every:
            checkpoint_callback(self.model, step)
        return self.model, self.train_loss_history

    # Get loss during training
    def get_erm_loss(self, model, X_train, Y_train):
        # Shift to evaluation mode
        model.eval()

        # Get ERM loss
        z = model(X_train)  # Get: [n_train, d_z]

        loss = 0

        if self.pro_params['problem_name'] == 'Newsvendor':
            loss = newsvendor_loss(z, Y_train, self.pro_params['h_cost'], self.pro_params['b_cost'])

        elif self.pro_params['problem_name'] == 'Portfolio':
            loss = get_loss(z, Y_train, self.pro_params)

        elif self.pro_params['problem_name'] == 'Inventory':
            loss = get_loss(z, Y_train, self.pro_params)

        return loss.mean()


    # Get loss for evaluation
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
                loss = get_loss(z, Y_eval, self.pro_params)

        elif self.pro_params['problem_name'] == 'Inventory':
            with torch.no_grad():
                loss = get_loss(z, Y_eval, self.pro_params)

         # Return mean loss as a tensor
        return loss.mean()

