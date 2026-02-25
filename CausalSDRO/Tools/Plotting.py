""""
    This file includes plotting functions for convergence, distribution visualization and bloxplots
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch

'''
    Draw convergence curve based on training history 
'''

def plot_convergence_curve(history_dict, output_dir, img_name):

    plt.figure(figsize=(10, 6))

    for model_name, loss_list in history_dict.items():
        if not loss_list:
            continue

        x_axis = [i * 5 for i in range(1, len(loss_list) + 1)]
        plt.plot(x_axis, loss_list, label=f'{model_name} (Eval Loss)', marker='o', markersize=4)

    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.title(f'Convergence Curve: {img_name}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.yscale('log')

    save_path = os.path.join(output_dir, f"{img_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Convergence plot saved to {save_path}")


'''
    Draw distribution fitting figures
'''

def plot_fit_visualization_newsvendor(data_gen, X_train, Y_train, models_dict, pro_params, output_dir):
    #  Plots the data distribution and fitted decision rules (Only for Newsvendor).

    p = pro_params['norm']
    l = pro_params['lambda']
    e = pro_params['epsilon']
    n = X_train.shape[0]
    d_x = X_train.shape[1]

    # (12,7) -> (25, 1)
    plt.figure(figsize=(12, 7))

    # Get true function and data
    f_true_func = data_gen.f_true
    beta_coeff = data_gen.BETA_COEFF

    # Project data to 1D
    X_train_proj = (X_train @ beta_coeff).cpu().numpy().squeeze()
    Y_train_np = Y_train.cpu().numpy().squeeze()

    # Sort for plotting lines
    sort_indices = np.argsort(X_train_proj)
    X_train_proj_sorted = X_train_proj[sort_indices]
    X_train_sorted = X_train[sort_indices]

    plt.xlabel('Projection $\mathbf{β}^{\mathrm{T}}  \mathbf{x}$', fontsize=18)
    plt.ylabel('Demand ($\mathbf{y}$) / Decision ($\mathbf{z}$)', fontsize=18)

    plt.tick_params(axis='both', which='major', labelsize=14)

    # zorder=5 (Highest layer)
    plt.scatter(X_train_proj, Y_train_np, alpha=0.2, label='Training Data Points ', s=10, zorder=5)

    # Plot true conditional mean and noise band
    with torch.no_grad():
        beta_x = X_train_sorted @ beta_coeff
        Y_true_mean = f_true_func(beta_x).cpu().numpy().squeeze()

    # zorder=4 (Second highest layer)
    plt.plot(X_train_proj_sorted, Y_true_mean, color='blue', linestyle='--', label='True Conditional Mean',
             zorder=4)
    # zorder=3 (Third highest layer, above fitted curves)
    plt.fill_between(X_train_proj_sorted, Y_true_mean - 1.0, Y_true_mean + 1.0,
                     color='blue', alpha=0.1, label='True Noise Band (±1 std)', zorder=3)

    # Plot fitted models from all methods
    for model_name, model in models_dict.items():
        with torch.no_grad():
            model.eval()
            decisions = model(X_train_sorted).cpu().numpy().squeeze()

        # Parse name to create the label
        rule_type = "SRF" if "SRF" in model_name else "2NN"

        # Determine algo_type and assign colors/styles

        if "SCSC" in model_name:
            algo_type = "SCSC"
            if rule_type == "2NN":
                color = "darkgreen"
                linestyle = "-"
            else:  # SRF
                color = "darkorange"
                linestyle = "-"

            label = f'{rule_type} Decision Rule'

            # Plot SCSC
            plt.plot(X_train_proj_sorted, decisions,
                     color=color,
                     linestyle=linestyle,
                     label=label,
                     alpha=0.8,
                     linewidth=2.5,
                     zorder=2)

    plt.legend(fontsize=14, loc='upper right')
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.tight_layout()

    # Save plot
    filename = f"distribution_fit_p{p}_l{l}_e{e}_N{n}_D{d_x}"

    # Save in multiple formats.
    base_path = os.path.join(output_dir, filename)
    try:
        plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.savefig(f"{base_path}.pdf", format='pdf', bbox_inches='tight', pad_inches=0.02)
    except Exception as e:
        print(f"Warning: Failed to save plot {filename}. Error: {e}")

    plt.close()