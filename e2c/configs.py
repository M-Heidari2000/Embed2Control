from dataclasses import dataclass, asdict

@dataclass
class TrainConfig:
    seed: int = 0
    log_dir: str = 'log'
    test_interval: int = 10
    action_repeat: int = 1
    state_dim: int = 30
    hidden_dim: int = 32
    min_var: float = 1e-2
    dropout_p: float = 0.01
    buffer_capacity: int = 1000000
    all_episodes: int = 1000
    seed_episodes: int = 5
    collect_interval: int = 100
    batch_size: int = 50
    lr: float = 1e-3
    eps: float = 1e-5
    clip_grad_norm: int = 1000
    free_nats: int = 0
    kl_beta: float = 1
    planning_horizon: int = 12
    num_candidates: int = 1000
    action_noise_var: float = 0.3

    dict = asdict