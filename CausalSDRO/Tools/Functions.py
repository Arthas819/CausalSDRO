"""
    This file includes some calculation functions.
"""
import math
import torch.nn as nn

from CausalSDRO.Optimizers.Gurobi_Optimizer import *

'''
    Get loss value. 
'''
def get_loss(z, y_train, pro_params):
    loss = 0
    if pro_params['problem_name'] == 'Newsvendor':
        loss = newsvendor_loss(z, y_train, pro_params['h_cost'], pro_params['b_cost'])

    elif pro_params['problem_name'] == 'Portfolio':
        d_z = pro_params['dimension'][2]
        z_decision, beta = parse_mv_output(z, d_z)
        loss = portfolio_loss_mv(z_decision, y_train, beta, pro_params['eta'])

    elif pro_params['problem_name'] == 'Inventory':
        loss = DifferentiableGurobiSolver.apply(z, y_train, pro_params)

    return loss


'''
    Loss calculator for testing set. 
'''
def get_test_loss(model, X_test, Y_test, pro_params):

    # pro_params is a dict sorting the problem paramters
    d_z = pro_params['dimension'][2]

    # Shift to evaluation mode
    model.eval()

    if pro_params['problem_name'] == 'Newsvendor':
        # Get ERM loss
        with torch.no_grad():
            z = model(X_test).reshape(-1, d_z) # Get: [n_train, d_z]
            loss = newsvendor_loss(z, Y_test, pro_params['h_cost'], pro_params['b_cost'])
            # Return mean loss as a tensor
            return loss.mean().item()

    elif pro_params['problem_name'] == 'Portfolio':
        with torch.no_grad():
            z = model(X_test).reshape(-1, d_z)  # Get: [n_train, d_z]
            z_decision, beta = parse_mv_output(z, d_z)
            # loss = portfolio_loss_mv(z_decision, Y_test, beta, pro_params['eta'])
            # Calculate real loss
            z_decision_w = z_decision.detach().cpu().numpy()
            return_value  = (pro_params['future_return'] * z_decision_w).sum(axis=1)
            loss, _, _, _, _ = metrics(return_value, pro_params['eta'])
            # Return mean loss as a tensor
            return loss.mean().item()

    elif pro_params['problem_name'] == 'Inventory':
        with torch.no_grad():
            z = model(X_test).reshape(-1, d_z)  # Get: [n_train, d_z]
            loss = DifferentiableGurobiSolver.apply(z, Y_test, pro_params)
            # Return mean loss as a tensor
            return loss.mean().item()

    else:
        return 0

'''
    Newsvendor loss
'''

def newsvendor_loss(z, y, h_cost, b_cost):
    # z: [batch_size, d_z], decision
    # y: [batch_size, d_y], demand
    m = nn.Softplus(beta=5)
    holding  = h_cost * m(z - y)
    stockout = b_cost * m(y - z)
    loss = holding + stockout
    # Return total cost
    # We need to check the dimension of loss, when d_z > 1, loss.sum(...)
    return loss

'''
    Portfolio loss
    Input: decision z, future return rate y, expected return beta, and weight eta
'''

def portfolio_loss_mv(z, y, beta, eta):
    # calculate total return value
    port_return = (y * z).sum(dim=1, keepdim=True)
    # Loss calculation
    loss = (port_return - beta) ** 2 - eta * port_return
    # return: [batch]
    return loss.squeeze()


'''
   [Portfolio] Get decision z and variance beta from the output of models (d_z = d_y + 1)
'''

def parse_mv_output(output, d_z):
    # Get portfolio decision z (outout: [batch, d_z])
    z_logits = output[..., : d_z - 1]
    # Use Normalization to satisfy the constraints
    sum_val = torch.sum(z_logits, dim=-1, keepdim=True)
    z = z_logits / (sum_val + 1e-8)
    # Estimated expected return
    beta = output[..., d_z - 1:]

    return z, beta


'''
    [Portfolio] Calculate the return of a decision in serval days,
    including mean, standard derivation, shape, and loss 
'''

def metrics(ret, eta):
    mean, std = ret.mean(), ret.std(ddof=1)
    sharpe = np.sqrt(252) * mean / std if std else np.nan

    losses = -ret
    var_loss = np.percentile(losses, 95)
    cvar = losses[losses >= var_loss].mean()

    var_term  = ((ret - mean)**2).sum() / 60
    mean_term = -eta * mean
    loss = var_term + mean_term

    return loss, sharpe, cvar, std, mean


'''
    [Inventory] Calculate the optimal value F* for the inventory problem
'''

def calculate_F_star_theoretical_is(data_gen, X_test, Y_test, pro_params):

    # n_x
    N_oracle_x = 1000
    # Use SAA to approximate the F*, M_oracle_y is the number of sample
    M_oracle_y = 100
    N_test_y = Y_test.shape[0]

    oracle_losses = []

    # Get costs
    h = pro_params['h_cost'].cpu().numpy()
    b = pro_params['b_cost'].cpu().numpy()
    c = pro_params['c_cost'].cpu().numpy()
    S = pro_params['s_cost'].cpu().numpy()
    d_z = len(c)
    d_y = len(b)
    d_x = X_test.shape[1]

    X_oracle = X_test  # [N_x, d_x]


    for i in range(N_oracle_x):

        x_i = X_oracle[i]  # [d_x]

        lambda_val = x_i @ data_gen.BETA_COEFF
        Y_j_list = []
        for _ in range(M_oracle_y):
            Y_j_list.append(data_gen.f_true(lambda_val))
        Y_for_saa = torch.stack(Y_j_list)  # [M_y, d_y]

        m_oracle = gp.Model(f'Oracle_SAA_x_{i}')
        m_oracle.Params.OutputFlag = 0

        # Decision variable z (inventory)
        z = m_oracle.addVars(d_z, lb=0.0, name='z')

        # Second-stage variables for *each* scenario j
        w = m_oracle.addVars(M_oracle_y, d_z, d_y, lb=0.0, name='w')  # w_j,i,k
        u = m_oracle.addVars(M_oracle_y, d_z, lb=0.0, name='u')  # u_j,i
        u_prime = m_oracle.addVars(M_oracle_y, d_y, lb=0.0, name='u_prime')  # u'_j,k

        # Second-stage objective terms
        subst_cost = (1 / M_oracle_y) * gp.quicksum(S[i][k] * w[j, i, k]
                                                    for j in range(M_oracle_y)
                                                    for i in range(d_z) for k in range(i, d_y))
        hold_cost = (1 / M_oracle_y) * gp.quicksum(h[i] * u[j, i]
                                                   for j in range(M_oracle_y) for i in range(d_z))
        short_cost = (1 / M_oracle_y) * gp.quicksum(b[k] * u_prime[j, k]
                                                    for j in range(M_oracle_y) for k in range(d_y))

        total_objective = gp.LinExpr()
        total_objective.add(gp.quicksum(c[i] * z[i] for i in range(d_z)))
        total_objective.add(subst_cost)
        total_objective.add(hold_cost)
        total_objective.add(short_cost)
        m_oracle.setObjective(
            total_objective,
            GRB.MINIMIZE
        )

        # Second-stage constraints for *each* scenario j
        Y_j_numpy = Y_for_saa.cpu().numpy()
        for j in range(M_oracle_y):
            y_j = Y_j_numpy[j]
            m_oracle.addConstrs((gp.quicksum(w[j, i, k] for k in range(i, d_y)) + u[j, i] == z[i]
                                 for i in range(d_z)), name=f'inv_bal_{j}')
            m_oracle.addConstrs((gp.quicksum(w[j, i, k] for i in range(k + 1)) + u_prime[j, k] - y_j[k] == 0
                                 for k in range(d_y)), name=f'dem_bal_{j}')

        m_oracle.optimize()

        oracle_losses.append(m_oracle.ObjVal)

    # Final F* is the average over all x_i
    F_star_approx = np.mean(oracle_losses) - 1 / math.sqrt(M_oracle_y)

    return F_star_approx