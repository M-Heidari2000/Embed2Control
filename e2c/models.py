import torch
import torch.nn as nn
from typing import Optional
from torch.distributions import MultivariateNormal
import einops


class Encoder(nn.Module):
    """
        p(z|x)
    """

    def __init__(
        self,
        observation_dim: int,
        state_dim: int,
        hidden_dim: Optional[int]=None,
        min_var: Optional[float]=1e-3,
        dropout_p: Optional[float]=0.4,
    ):
        super().__init__()

        hidden_dim = hidden_dim if hidden_dim is not None else 2*observation_dim

        self.mlp_layers = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
        )

        self.mean_head = nn.Linear(hidden_dim, state_dim)
        self.var_head = nn.Sequential(
            nn.Linear(hidden_dim, state_dim),
            nn.Softplus(),
        )

        self._min_var = min_var

    def forward(self, observation):
        hidden = self.mlp_layers(observation)
        mean = self.mean_head(hidden)
        var = self.var_head(hidden) + self._min_var

        return MultivariateNormal(mean, torch.diag_embed(var))
    

class Decoder(nn.Module):
    """
        p(x|z)
    """

    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        hidden_dim: Optional[int]=None,
        dropout_p: Optional[float]=1e-3,
    ):
        
        super().__init__()
        
        hidden_dim = hidden_dim if hidden_dim is not None else 2*state_dim

        self.mlp_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, observation_dim),
        )

    
    def forward(self, state):
        return self.mlp_layers(state)
    

class TransitionModel(nn.Module):

    """
        Estimates the locally linear dynamics matrices for a given latent z
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: Optional[int]=None,
        min_var: Optional[float]=1e-3,
        dropout_p: Optional[float]=0.4,
    ):
        
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim

        hidden_dim = hidden_dim if hidden_dim is not None else 2*state_dim

        self.mlp_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
        )

        # A = I + v @ r.T
        self.v_head = nn.Linear(hidden_dim, state_dim)
        self.r_head = nn.Linear(hidden_dim, state_dim)

        self.B_head = nn.Linear(hidden_dim, state_dim * action_dim)

        # offset
        self.o_head = nn.Linear(hidden_dim, state_dim)

        # system noise
        self.w_head = nn.Sequential(
            nn.Linear(hidden_dim, state_dim),
            nn.Softplus(),
        )

        self._min_var = min_var

    def forward(
        self,
        state_sample,
        action,
        state_dist,
        ):

        hidden = self.mlp_layers(state_sample)
        v = self.v_head(hidden)
        v = einops.rearrange(v, "b s -> b s 1")
        r = self.r_head(hidden)
        r = einops.rearrange(r, "b s -> b 1 s")
        A = (
            torch.eye(self.state_dim, device=state_sample.device).repeat(r.shape[0], 1, 1) + torch.bmm(v, r)
        )
        B = self.B_head(hidden)
        B = einops.rearrange(B, "b (s a) -> b s a", s=self.state_dim, a=self.action_dim)
        o = self.o_head(hidden)
        w = self.w_head(hidden) + self._min_var

        # at this point 
        # A: b * s * s
        # B: b * s * u
        # o: b * s
        
        # next state mean computation
        mu = einops.rearrange(state_dist.loc, "b s -> b s 1")
        action = einops.rearrange(action, "b a -> b a 1")
        o = einops.rearrange(o, "b s -> b s 1")
        next_state_mean = torch.bmm(A, mu) + torch.bmm(B, action) + o
        next_state_mean = einops.rearrange(next_state_mean, "b s 1 -> b s")

        # next state covariance computation
        H = torch.diag_embed(w)    # b * s * s
        sigma = state_dist.covariance_matrix    # b * s * s

        C = H + torch.bmm(
            torch.bmm(A, sigma),
            A.transpose(1, 2)
        )

        next_state_dist = MultivariateNormal(next_state_mean, C)

        return next_state_dist