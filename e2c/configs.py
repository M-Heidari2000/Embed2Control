from dataclasses import dataclass, asdict

@dataclass
class TrainConfig:
    seed: int = 0
    log_dir: str = 'log'
    state_dim: int = 30
    hidden_dim: int = 32
    min_var: float = 1e-2
    dropout_p: float = 0.01
    buffer_capacity: int = 1000000
    num_episodes: int = 100
    num_epochs: int = 1024
    batch_size: int = 50
    lr: float = 1e-3
    eps: float = 1e-5
    clip_grad_norm: int = 1000
    free_nats: int = 0
    kl_beta: float = 1

    dict = asdict