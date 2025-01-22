import torch


class Quadratic:

    """
        c(x, u) = 0.5 * (x-x*).T @ Q @ (x-x*) + 0.5 * a.T @ R @ a
    """

    def __init__(self, Q, R, target, device: str="cpu"):
        """
            Q: s*s
            R: u*u
            x_target: 1 * s
        """

        self.device = device
        self.Q = torch.as_tensor(Q, device=self.device, dtype=torch.float32)
        self.R = torch.as_tensor(R, device=self.device, dtype=torch.float32)
        self.target = torch.as_tensor(target, device=self.device, dtype=torch.float32)

    
    def __call__(self, state, action):
        """
            state: b * s
            action: b * a
        """
        state = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        action = torch.as_tensor(action, device=self.device, dtype=torch.float32)

        cost = 0.5 * (state - self.target) @ self.Q @ (state - self.target).T + 0.5 * action @ self.R @ action.T
        return cost.diag()