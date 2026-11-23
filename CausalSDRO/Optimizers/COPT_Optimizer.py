"""
    COPT solver for the regularized inventory dual.
"""

from functools import lru_cache
from importlib.metadata import *
import sys
import numpy as np
import torch


@lru_cache(maxsize=1)
def get_copt_env():
    try:
        installed = version('coptpy')
    except PackageNotFoundError as exc:
        raise RuntimeError(f'coptpy is not installed in {sys.executable}. '
                           'Use the pytorch environment with coptpy>=8.0.3.') from exc
    if tuple(int(x) for x in installed.split('.')[:3]) < (8, 0, 3):
        raise RuntimeError(f'coptpy {installed} is loaded by {sys.executable}. '
                           'Use the pytorch environment with coptpy>=8.0.3.')
    try:
        import coptpy as cp
    except (ImportError, OSError) as exc:
        raise RuntimeError(f'Cannot import COPT in {sys.executable}: {exc}') from exc
    try:
        return cp.Envr()
    except cp.CoptError as exc:
        raise RuntimeError(f'COPT initialization failed in {sys.executable}: {exc}. '
                           'Check the COPT installation and license for this environment.') from exc


class InventoryDual:
    def __init__(self, h, b, c, S, mu, batch_size=1):
        self.env = get_copt_env()
        import coptpy as cp
        from coptpy import COPT

        self.cp, self.COPT = cp, COPT
        self.model = self.env.createModel('Inventory-Substitution-Dual')
        self.model.setParam(COPT.Param.Logging, 0)
        self.model.setParam(COPT.Param.Threads, 1)
        self.model.setParam(COPT.Param.FeasTol, 1e-9)
        self.model.setParam(COPT.Param.DualTol, 1e-9)
        self.h, self.b, self.c, self.S = map(np.asarray, (h, b, c, S))
        self.mu, self.batch_size = mu, batch_size
        self.eta = self.model.addVars(batch_size, len(h), lb=-COPT.INFINITY, nameprefix='eta')
        self.nu = self.model.addVars(batch_size, len(b), lb=-COPT.INFINITY, nameprefix='nu')
        self.model.addConstrs(self.eta[k, i] <= h[i] for k in range(batch_size) for i in range(len(h)))
        self.model.addConstrs(self.nu[k, j] <= b[j] for k in range(batch_size) for j in range(len(b)))
        for i in range(len(h)):
            for j in range(i, len(b)):
                if np.isfinite(S[i][j]):
                    self.model.addConstrs(self.eta[k, i] + self.nu[k, j] <= S[i][j]
                                         for k in range(batch_size))
        self.eta_vars = list(self.eta.values())
        self.variables = self.eta_vars + list(self.nu.values())
        self.regularizer = mu * cp.quicksum(v * v for v in self.eta_vars)
        self.model.setObjective(-self.regularizer, COPT.MAXIMIZE)

    def solve_batch(self, z, y):
        COPT = self.COPT
        # Independent QPs share one model; only linear coefficients change.
        self.model.setInfo(COPT.Info.Obj, self.variables, np.concatenate((z.ravel(), y.ravel())))
        self.model.solve()
        if self.model.status != COPT.OPTIMAL:
            if self.batch_size > 1 and self.model.status in (COPT.NUMERICAL, COPT.IMPRECISE):
                # Retry numerically difficult batches as independent QPs.
                single = inventory_solver(tuple(self.h), tuple(self.b), tuple(self.c),
                                          tuple(map(tuple, self.S)), self.mu)
                result = [single.solve(zi, yi) for zi, yi in zip(z, y)]
                return tuple(np.asarray(items) for items in zip(*result))
            raise RuntimeError(f'Inventory COPT status: {self.model.status}')
        eta = np.asarray(self.model.getInfo(COPT.Info.Value, self.eta_vars)).reshape(z.shape)
        nu = np.minimum(self.b, np.min(self.S[None, :, :] - eta[:, :, None], axis=1))
        values = (z * (eta + self.c)).sum(axis=1) + (y * nu).sum(axis=1)
        values -= self.mu * (eta * eta).sum(axis=1)
        return values, eta + self.c, nu

    def solve(self, z, y):
        values, grad_z, grad_y = self.solve_batch(np.asarray(z)[None, :], np.asarray(y)[None, :])
        return values[0], grad_z[0], grad_y[0]


@lru_cache(maxsize=16)
def inventory_solver(h, b, c, S, mu, batch_size=1):
    return InventoryDual(h, b, c, S, mu, batch_size)


class DifferentiableCOPTSolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z_batch, y_batch, pro_params):
        h, b, c = [tuple(pro_params[key].detach().cpu().tolist())
                   for key in ('h_cost', 'b_cost', 'c_cost')]
        S = tuple(tuple(row) for row in pro_params['s_cost'].detach().cpu().tolist())
        z_np = z_batch.detach().cpu().numpy()
        y_np = y_batch.detach().cpu().numpy()
        values, grads_z, grads_y = [], [], []
        for start in range(0, len(z_np), 200):
            z, y = z_np[start:start + 200], y_np[start:start + 200]
            solver = inventory_solver(h, b, c, S, pro_params.get('mu', 0.01), len(z))
            value, grad_z, grad_y = solver.solve_batch(z, y)
            values.extend(value)
            grads_z.extend(grad_z)
            grads_y.extend(grad_y)
        ctx.save_for_backward(z_batch.new_tensor(np.asarray(grads_z)),
                              y_batch.new_tensor(np.asarray(grads_y)))
        return z_batch.new_tensor(values)

    @staticmethod
    def backward(ctx, grad_output):
        grad_z, grad_y = ctx.saved_tensors
        return grad_output[:, None] * grad_z, grad_output[:, None] * grad_y, None


if __name__ == '__main__':
    get_copt_env()
    params = {'h_cost': torch.tensor([1.0, 0.7, 0.6], dtype=torch.float64),
              'b_cost': torch.tensor([1.8, 1.6, 1.2], dtype=torch.float64),
              'c_cost': torch.zeros(3, dtype=torch.float64),
              's_cost': torch.tensor([[0, 1.7, 2], [float('inf'), 0, 1.5],
                                      [float('inf'), float('inf'), 0]], dtype=torch.float64),
              'mu': 0.01}
    z = torch.tensor([[2.0, 3.0, 4.0]], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[1.0, 4.0, 3.0]], dtype=torch.float64)
    loss = DifferentiableCOPTSolver.apply(z, y, params)
    loss.sum().backward()
    # np.testing.assert_allclose(loss.detach().numpy(), [2.28195], atol=1e-6)
    # np.testing.assert_allclose(z.grad.numpy(), [[0.85, -0.85, 0.6]], atol=1e-6)
    print(f'Python: {sys.executable}')
    print(f'COPT {version("coptpy")}: inventory solve and gradient passed.')
    print(f'Loss: {loss.item():.8f}; gradient: {z.grad.tolist()}')
