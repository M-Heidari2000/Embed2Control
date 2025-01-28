import json
import torch
import numpy as np
from pathlib import Path
import gymnasium as gym
import matplotlib.pyplot as plt
from e2c.agent import ILQRAgent
from e2c.models import (
    Encoder,
    Decoder,
    TransitionModel,
)
from e2c.configs import TrainConfig
from e2c.cost_functions import Quadratic
from gymnasium.wrappers import RescaleAction
from matplotlib import animation


def save_frames_as_gif(frames, file_name: str):
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(1, 1, 1)
    
    patch = ax.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(
        fig, animate, frames = len(frames), interval=50
    )
    FFwriter = animation.FFMpegWriter(fps=10)
    anim.save(filename=file_name, writer=FFwriter)


# load the environment
env = gym.make("Pendulum-v1", render_mode="rgb_array")
# the training code assumes actions are normalized in (-1, 1)
# if the environment does not satisfy this (the Pendulum-v1 does not), use a wrapper
env = RescaleAction(env=env, min_action=-1.0, max_action=1.0)

model_dir = Path("log/20250126_1727")

with open(model_dir / "args.json", "r") as f:
    config = TrainConfig(**json.load(f))

device = "cuda" if torch.cuda.is_available() else "cpu"

encoder = Encoder(
    state_dim=config.state_dim,
    observation_dim=env.observation_space.shape[0],
    hidden_dim=config.hidden_dim,
    min_var=config.min_var,
    dropout_p=config.dropout_p,
).to(device)

decoder = Decoder(
    state_dim=config.state_dim,
    observation_dim=env.observation_space.shape[0],
    hidden_dim=config.hidden_dim,
    dropout_p=config.dropout_p,
).to(device)

transition_model = TransitionModel(
    state_dim=config.state_dim,
    action_dim=env.action_space.shape[0],
    hidden_dim=config.hidden_dim,
    min_var=config.min_var,
    dropout_p=config.dropout_p,
).to(device)


# load state dicts
encoder.load_state_dict(torch.load(model_dir / "encoder.pth", weights_only=True))
decoder.load_state_dict(torch.load(model_dir / "decoder.pth", weights_only=True))
transition_model.load_state_dict(torch.load(model_dir / "transition_model.pth", weights_only=True))

encoder.eval()
decoder.eval()
transition_model.eval()

# specify the target in the observation space
target = np.array([1, 0, 0], dtype=np.float32)
with torch.no_grad():
    obs_target = torch.as_tensor(target, device=device, dtype=torch.float32).unsqueeze(0)
    state_target = encoder(obs_target).loc

# define the cost function
cost_function = Quadratic(
    Q=torch.eye(config.state_dim, device=device, dtype=torch.float32),
    R=torch.eye(env.action_space.shape[0], device=device, dtype=torch.float32),
    target=state_target,
    device=device,
)

agent = ILQRAgent(
    encoder=encoder,
    transition_model=transition_model,
    cost_function=cost_function,
    planning_horizon=4,
    num_iterations=10,
    sample=False,
)

# test on one episode
obs, _ = env.reset()
total_reward = 0
done = False
frames = []

while not done:
    frames.append(env.render())
    planned_actions = agent(obs=obs)
    action = planned_actions[0]
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

# save the animation
save_frames_as_gif(frames=frames, file_name="animation1.mp4")