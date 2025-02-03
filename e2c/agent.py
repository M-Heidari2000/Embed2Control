import numpy as np
import torch
import einops
from torch.distributions import Uniform


class RSAgent:
    """
        action planning by Random Shooting
    """
    def __init__(
        self,
        encoder,
        transition_model,
        cost_function,
        planning_horizon: int,
        num_candidates: int,
    ):
        self.encoder = encoder
        self.transition_model = transition_model
        self.cost_function = cost_function
        self.num_candidates = num_candidates
        self.planning_horizon = planning_horizon

        self.device = next(encoder.parameters()).device

    def __call__(self, obs):

        # convert o_t to a torch tensor and add a batch dimension
        obs = torch.as_tensor(obs, device=self.device).repeat(self.num_candidates, 1)

        # no learning takes place here
        with torch.no_grad():
            self.encoder.eval()
            self.transition_model.eval()

            state_dist = self.encoder(obs)

            action_dist = Uniform(
                low=-torch.ones((self.planning_horizon, self.transition_model.action_dim), device=self.device),
                high=torch.ones((self.planning_horizon, self.transition_model.action_dim),device=self.device),
            )

            action_candidates = action_dist.sample([self.num_candidates])
            action_candidates = einops.rearrange(action_candidates, "n h a -> h n a")

            state = state_dist.sample()
            total_predicted_reward = torch.zeros(self.num_candidates, device=self.device)

            # start generating trajectories starting from s_t using transition model
            for t in range(self.planning_horizon):
                total_predicted_reward += self.cost_function(
                    state=state,
                    action=action_candidates[t],
                ).squeeze()
                # get next state from our prior (transition model)
                state_dist, _, _, _ = self.transition_model(
                    state,
                    action_candidates[t],
                    state_dist,
                )
                state = state_dist.sample()

            # find the best action sequence
            min_index = total_predicted_reward.argmin()
            actions = action_candidates[:, min_index, :]

        return actions.cpu().numpy()
    

class ILQRAgent:
    """
        action planning by the IQLR method
    """
    def __init__(
        self,
        encoder,
        transition_model,
        cost_function,
        planning_horizon: int,
        num_iterations: int,
        sample: bool=True,
    ):
        self.encoder = encoder
        self.transition_model = transition_model
        self.cost_function = cost_function
        self.num_iterations = num_iterations
        self.planning_horizon = planning_horizon
        self.sample = sample

        self.device = next(encoder.parameters()).device

    def __call__(self, obs):

        # convert o_t to a torch tensor and add a batch dimension
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)

        # no learning takes place here
        with torch.no_grad():
            self.encoder.eval()
            self.transition_model.eval()

            target = self.cost_function.target

            # initial policy (zero actions)
            Ks = [
                torch.zeros(
                    (self.transition_model.action_dim, self.transition_model.state_dim),
                    device=self.device,
                    dtype=torch.float32,
                )
            ] * self.planning_horizon

            ks = [
                torch.zeros(
                    (self.transition_model.action_dim, 1),
                    device=self.device,
                    dtype=torch.float32,
                )
            ] * self.planning_horizon
            state_dist = self.encoder(obs)

            for _ in range(self.num_iterations + 1):
                state_dist = self.encoder(obs)

                As = []
                Bs = []
                os = []
                actions = torch.zeros(
                    (self.planning_horizon, self.transition_model.action_dim),
                    device=self.device,
                    dtype=torch.float32,
                )
                # rollout a trajectory with current policy
                for t in range(self.planning_horizon):
                    state = state_dist.sample() if self.sample else state_dist.loc
                    action = (state - target) @ Ks[t].T + ks[t].T
                    actions[t] = action
                    state_dist, A, B, o = self.transition_model(
                        state_sample=state,
                        action=action,
                        state_dist=state_dist,
                    )
                    As.append(A.squeeze(0))
                    Bs.append(B.squeeze(0))
                    os.append(o.squeeze(0))
                # compute a new policy
                Ks, ks = self._compute_policy(
                    As=As,
                    Bs=Bs,
                    os=os,
                )

        return np.clip(actions.cpu().numpy(), min=-1.0, max=1.0)
    
    def _compute_policy(self, As, Bs, os):
        state_dim, action_dim = Bs[0].shape

        Ks = []
        ks = []

        V = torch.zeros((state_dim, state_dim), device=self.device)
        v = torch.zeros((state_dim, 1), device=self.device)

        C = torch.block_diag(self.cost_function.Q, self.cost_function.R)
        c = torch.zeros((state_dim + action_dim, 1), device=self.device)

        for t in range(self.planning_horizon-1, -1, -1):
            F = torch.concatenate((As[t], Bs[t]), dim=1)
            f = os[t] + (As[t] - torch.eye(state_dim, device=self.device)) @ self.cost_functioncost_function.target.T
            Q = C + F.T @ V @ F
            q = c + F.T @ V @ f + F.T @ v
            Qxx = Q[:state_dim, :state_dim]
            Qxu = Q[:state_dim, state_dim:]
            Qux = Q[state_dim:, :state_dim]
            Quu = Q[state_dim:, state_dim:]
            qx = q[:state_dim, :]
            qu = q[state_dim:, :]

            K = - torch.linalg.pinv(Quu) @ Qux
            k = - torch.linalg.pinv(Quu) @ qu
            V = Qxx + Qxu @ K + K.T @ Qux + K.T @ Quu @ K
            v = qx + Qxu @ k + K.T @ qu + K.T @ Quu @ k

            Ks.append(K)
            ks.append(k)
        
        return Ks[::-1], ks[::-1]