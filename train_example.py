import gymnasium as gym
from e2c.configs import TrainConfig
from e2c.train import train
from gymnasium.wrappers import RescaleAction


# load the environment
env = gym.make("Pendulum-v1", render_mode="rgb_array")
# the training code assumes actions are normalized in (-1, 1)
# if the environment does not satisfy this (the Pendulum-v1 does not), use a wrapper
env = RescaleAction(env=env, min_action=-1.0, max_action=1.0)

train_config = TrainConfig(
    state_dim=3,
    hidden_dim=64,
    num_episodes=20,
    num_epochs=20,
    free_nats=2,
)

results = train(env=env, config=train_config)

print(results)