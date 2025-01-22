import torch
import torch.nn as nn
from typing import Optional
from torch.distributions import Normal
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
        min_std: Optional[float]=1e-3,
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
        self.std_head = nn.Sequential(
            nn.Linear(hidden_dim, state_dim),
            nn.Softplus(),
        )

        self._min_std = min_std

    def forward(self, observation):
        hidden = self.mlp_layers(observation)
        mean = self.mean_head(hidden)
        std = self.std_head(hidden) + self._min_std

        return Normal(mean, std)
    

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
        hidden_dim: Optional[int],
        min_std: Optional[float]=1e-3,
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

        self._min_std = min_std

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
        B = einops.rearrange(B, "b (s u) -> b s u", s=self.state_dim, u=self.action_dim)
        o = self.o_head(hidden)
        w = self.w_head(hidden) + self._min_std

        # at this point 
        # A: b * s * s
        # B: b * s * u
        # o: b * s
        
        # next state mean computation
        mu = einops.rearrange(state_dist.loc, "b s -> b s 1")
        action = einops.rearrange(action, "b u -> b u 1")
        o = einops.rearrange(o, "b s -> b s 1")
        next_state_mean = torch.bmm(A, mu) + torch.bmm(B, action) + o
        next_state_mean = einops.rearrange(next_state_mean, "b s 1 -> b s")

        # next state covariance computation
        H = torch.diag_embed(w ** 2)    # b * s * s
        sigma = torch.diag_embed(state_dist.scale ** 2)    # b * s * s
        C = H + torch.bmm(
            torch.bmm(A, sigma),
            A.transpose(1, 2)
        )
        next_state_std = torch.sqrt(C.diagonal(dim1=1, dim2=2) + self._min_std)    # b * s

        next_state_dist = Normal(next_state_mean, next_state_std)

        return next_state_dist