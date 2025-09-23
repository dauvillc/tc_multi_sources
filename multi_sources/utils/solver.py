"""Implements classes to solve the flow matching ODE for multi-sources data."""

import torch


class MultisourceEulerODESolver:
    """Solves the ODE (df/dt)(x, t) = u(x, t) with the Euler method.
    u should receive the following arguments:
        - x: dict {source: x_s} where x_s is a tensor of shape (B, C, ...)
            giving the values of the flow for the source s at t;
        - t: tensor of shape (B,) giving the time at which the vector field is
            evaluated.
    u should return a dict {source: u_s} where u_s is a tensor of shape (B, C, ...)
    giving the values of the vector field for the source s at t.
    """

    def __init__(self, vf_func):
        """
        Args:
            vf_func (Callable): Function that computes the vector field u(x, t).
        """
        self.vf_func = vf_func

    def solve(self, x_0, time_grid, return_intermediate_steps=False):
        """Solves the ODE for the given initial conditions.
        Args:
            x_0 (dict): Initial conditions, dict {source: x_s} where x_s is a tensor
                of shape (B, C, ...) giving the initial values of the flow for the
                source s.
            time_grid (tensor): Time grid of shape (T,) within [0, 1]; times at which
                the flow will be evaluated. Also defines the step size for the Euler
                method.
            return_intermediate_steps (bool): If True, returns the intermediate solutions at
                each time step.
        Returns:
            sol: dict {source: x_s} where x_s is a tensor of shape (B, C, ...)
                giving the values of the flow for the source s at the times in time_grid.
                If return_intermediate is True, the values are stored in a tensor of shape
                (T, B, C, ...).
        """
        device = next(iter(x_0.values())).device
        time_grid = time_grid.to(device)
        x = {source: x_0[source].clone().to(device) for source in x_0}

        sol = {}
        # For each source, the returned solution will be a tensor of shape (T, B, C, ...)
        for source in x:
            sol[source] = torch.empty((len(time_grid), *x[source].shape), device=device)
            sol[source][0] = x[source]

        for k, (t0, t1) in enumerate(zip(time_grid[:-1], time_grid[1:])):
            u = self.vf_func(x, t0)
            for source in x:
                dt = t1 - t0
                x[source] = x[source] + dt * u[source]
                sol[source][k + 1] = x[source]

        if not return_intermediate_steps:
            sol = {source: sol[source][-1] for source in sol}
        return sol
