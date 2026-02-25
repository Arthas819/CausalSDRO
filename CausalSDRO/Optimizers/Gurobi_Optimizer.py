
"""
    This file includes a Gurobi optimizer with gradient for inventory problem
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import torch


class DifferentiableGurobiSolver(torch.autograd.Function):
    # This function calculate the results in a batch
    @staticmethod
    def forward(ctx, z_batch_tensor, y_batch_tensor, pro_params):
        # Forward: using Gurobi to solve the dual problem
        # z_batch_tensor: [BatchSize, d_z]
        # y_batch_tensor: [BatchSize, d_y]
        device = z_batch_tensor.device
        dtype = z_batch_tensor.dtype
        batch_size = z_batch_tensor.shape[0]

        # Transform tensor to numpy for Gurobi
        z_batch_np = z_batch_tensor.detach().cpu().numpy()
        y_batch_np = y_batch_tensor.detach().cpu().numpy()

        # Get parameters
        S = pro_params['s_cost'].cpu().numpy().tolist()
        h = pro_params['h_cost'].cpu().numpy().tolist()
        b = pro_params['b_cost'].cpu().numpy().tolist()
        c = pro_params['c_cost'].cpu().numpy().tolist()
        I = z_batch_tensor.shape[1]
        J = y_batch_tensor.shape[1]

        # Lists to store results from each sample in the batch
        obj_vals_list = []
        grads_z_list = []
        grads_y_list = []

        # --- For each batch ---
        for i in range(batch_size):
            z_i = z_batch_np[i]
            y_i = y_batch_np[i]
            # For all y_i[x] < 0 ,set them to 5
            y_i = np.where(y_i < 0, 5, y_i).tolist()
            # For all z_i[x] < 0 ,set them to 1
            z_i = np.where(z_i < 0, 1, z_i).tolist()

            m = gp.Model(f'Inventory-Substitution-Dual-{i}')
            m.Params.OutputFlag = 0

        #### ----------- Dual --------------
            eta = m.addVars(I, lb=-GRB.INFINITY, name='eta')
            nu = m.addVars(J, lb=-GRB.INFINITY, name='nu')

            m.addConstrs((eta[k] <= h[k] for k in range(I)), name='holding_bound')
            m.addConstrs((nu[k] <= b[k] for k in range(J)), name='shortage_bound')

            for k in range(I):
                for j in range(k, J):  #
                    m.addConstr(eta[k] + nu[j] <= S[k][j], name=f'sub_{k}_{j}')
            total_objective = gp.LinExpr()
            total_objective.add(gp.quicksum(y_i[j] * nu[j] for j in range(J)))
            total_objective.add(gp.quicksum(z_i[k] * (eta[k] + c[k]) for k in range(I)))
            m.setObjective(total_objective, GRB.MAXIMIZE)

            m.optimize()

        #### ----------- Primal --------------
            #
            # # Second-stage variables for *each* scenario j
            # w = m.addVars(I, J, lb=0.0, name='w')  # w_j,i,k
            # u = m.addVars(I, lb=0.0, name='u')  # u_j,i
            # u_prime = m.addVars(J, lb=0.0, name='u_prime')  # u'_j,k
            #
            # dual_1 = m.addConstrs((gp.quicksum(w[i, k] for k in range(i, J)) + u[i] == z_i[i]
            #                      for i in range(I)), name=f'inv_bal')
            # dual_2 = m.addConstrs((gp.quicksum(w[i, j] for i in range(j + 1)) + u_prime[j] - y_i[j] == 0
            #                      for j in range(J)), name=f'dem_bal')
            #
            # # Second-stage objective terms
            # subst_cost = gp.quicksum(S[i][k] * w[i, k] for i in range(I) for k in range(i, J))
            # hold_cost = gp.quicksum(h[i] * u[i] for i in range(I))
            # short_cost = gp.quicksum(b[k] * u_prime[k] for k in range(J))
            #
            # total_objective = gp.LinExpr()
            # total_objective.add(subst_cost)
            # total_objective.add(hold_cost)
            # total_objective.add(short_cost)
            # m.setObjective(total_objective, GRB.MINIMIZE)
            # m.optimize()
            # eta = [dual_1[k].Pi for k in range(I)]
            # nu = [dual_2[j].Pi for j in range(J)]

            # Store results for sample i
            if m.status == GRB.OPTIMAL:
                # print('Obj:', m.ObjVal)
                obj_vals_list.append(m.ObjVal)
                grads_z_list.append([eta[k].X + c[k] for k in range(I)])
                grads_y_list.append([nu[j].X for j in range(J)])
            else:
                print('Failed...')
                # Handle infeasibility, e.g., return 0s
                obj_vals_list.append(0.0)
                grads_z_list.append([c[k] for k in range(I)])  # Gradient is just c_i
                grads_y_list.append([0.0 for j in range(J)])

            # Convert lists of results back to tensors
        obj_vals_tensor = torch.tensor(obj_vals_list, dtype=dtype, device=device)
        grads_z_tensor = torch.tensor(grads_z_list, dtype=dtype, device=device)
        grads_y_tensor = torch.tensor(grads_y_list, dtype=dtype, device=device)

        # 4. Save gradients for backward pass
        ctx.save_for_backward(grads_z_tensor, grads_y_tensor)

        # 5. Return batch of loss values
        return obj_vals_tensor  # Shape: [BatchSize]

    @staticmethod
    def backward(ctx, grad_output):
        ## Backpropagation

        # grad_output: gradient from the upstream
        grad_z, grad_y = ctx.saved_tensors

        grad_output_expanded = grad_output.unsqueeze(1)

        # --- The Chain Rule ---
        return (grad_output_expanded * grad_z,
                grad_output_expanded * grad_y,
                None)  # pro_params need not gradient