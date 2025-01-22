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
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)

        # no learning takes place here
        with torch.no_grad():
            state_dist = self.encoder(obs)

            action_dist = Uniform(
                low=-torch.ones((self.planning_horizon, self.posterior_model.action_dim), device=self.device),
                high=torch.ones((self.planning_horizon, self.posterior_model.action_dim),device=self.device),
            )

            action_candidates = action_dist.sample([self.num_candidates])
            action_candidates = einops.rearrange(action_candidates, "n h a -> h n a")

            state = state_dist.sample([self.num_candidates]).squeeze(-2)
            total_predicted_reward = torch.zeros(self.num_candidates, device=self.device)

            # start generating trajectories starting from s_t using transition model
            for t in range(self.planning_horizon):
                total_predicted_reward += self.cost_function(state=state).squeeze()
                # get next state from our prior (transition model)
                state_dist = self.transition_model(
                    state,
                    action_candidates[t],
                    state_dist,
                )
                state = state_dist.sample()

            # find the best action sequence
            max_index = total_predicted_reward.argmax()
            actions = action_candidates[:, max_index, :]

        return actions.cpu().numpy()