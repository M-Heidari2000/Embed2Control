import os
import time
import json
import torch
import numpy as np
import gymnasium as gym
from pathlib import Path
from datetime import datetime
from torch.distributions.kl import kl_divergence
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard.writer import SummaryWriter
from .agent import RSAgent
from .memory import ReplayBuffer
from .configs import TrainConfig
from .models import (
    Encoder,
    Decoder,
    TransitionModel,
)
from .cost_functions import Quadratic
from torch.distributions import MultivariateNormal


def train(
    env: gym.Env,
    config: TrainConfig,
):

    # prepare logging
    log_dir = Path(config.log_dir) / datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs(log_dir, exist_ok=True)
    with open(log_dir / "args.json", "w") as f:
        json.dump(config.dict(), f)
    
    writer = SummaryWriter(log_dir=log_dir)

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # define replay buffer
    replay_buffer = ReplayBuffer(
        capacity=config.buffer_capacity,
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
    )

    # define models and optimizer
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

    all_params = (
        list(encoder.parameters()) +
        list(decoder.parameters()) +
        list(transition_model.parameters())
    )

    optimizer = torch.optim.Adam(all_params, lr=config.lr, eps=config.eps)

    # collect initial experience with random actions
    for episode in range(config.seed_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(
                observation=obs,
                action=action,
                next_observation=next_obs
            )
            obs = next_obs

    # main training loop
    for episode in range(config.seed_episodes, config.all_episodes):

        # update model parameters
        encoder.train()
        decoder.train()
        transition_model.train()

        start = time.time()
        for update_step in range(config.collect_interval):
            observations, actions, next_observations = replay_buffer.sample(
                batch_size=config.batch_size,
            )

            observations = torch.as_tensor(observations, device=device)
            actions = torch.as_tensor(actions, device=device)
            next_observations = torch.as_tensor(next_observations, device=device)

            priors = MultivariateNormal(
                torch.zeros((config.batch_size, config.state_dim), device=device, dtype=torch.float32),
                torch.diag_embed(torch.ones((config.batch_size, config.state_dim), device=device, dtype=torch.float32)),
            )
            posteriors = encoder(observations)
            posterior_samples = posteriors.rsample()
            next_priors = transition_model(
                state_sample=posterior_samples,
                action=actions,
                state_dist=posteriors
            )
            next_prior_samples = next_priors.rsample()
            next_posteriors = encoder(next_observations)

            kl = kl_divergence(posteriors, priors)
            kl_loss = kl.clamp(min=config.free_nats).mean()

            next_kl = kl_divergence(next_posteriors, next_priors)
            next_kl_loss = next_kl.clamp(min=config.free_nats).mean()

            recon_observations = decoder(posterior_samples)
            recon_next_observations = decoder(next_prior_samples)

            obs_loss = mse_loss(
                recon_observations,
                observations,
                reduction='none',
            ).mean([0, 1]).sum()

            next_obs_loss = mse_loss(
                recon_next_observations,
                next_observations,
                reduction='none',
            ).mean([0, 1]).sum()

            loss = obs_loss + next_obs_loss + config.kl_beta * next_kl_loss + kl_loss
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(all_params, config.clip_grad_norm)
            optimizer.step()

            # print losses and add tensorboard
            print('update_step: %3d loss: %.5f, kl_loss: %.5f, next_kl_loss: %.5f obs_loss: %.5f, next_obs_loss: %.5f'
                  % (update_step+1,
                     loss.item(), kl_loss.item(), next_kl_loss.item(), obs_loss.item(), next_obs_loss.item()))
            
            total_update_step = episode * config.collect_interval + update_step
            writer.add_scalar('overall loss', loss.item(), total_update_step)
            writer.add_scalar('kl loss', kl_loss.item(), total_update_step)
            writer.add_scalar('next_kl loss', next_kl_loss.item(), total_update_step)
            writer.add_scalar('obs loss', obs_loss.item(), total_update_step)
            writer.add_scalar('next obs loss', next_obs_loss.item(), total_update_step)
        
        print('elasped time for update: %.2fs' % (time.time() - start))

        # collect experience
        with torch.no_grad():
            encoder.eval()
            obs_target = torch.as_tensor(env.manifold(env.target).T, device=device, dtype=torch.float32)
            state_target = encoder(obs_target).loc

            cost_function = Quadratic(
                Q=torch.eye(config.state_dim, device=device, dtype=torch.float32),
                R=torch.eye(env.action_space.shape[0], device=device, dtype=torch.float32),
                target=state_target,
                device=device,
            )
        
        agent = RSAgent(
            encoder=encoder,
            transition_model=transition_model,
            cost_function=cost_function,
            planning_horizon=config.planning_horizon,
            num_candidates=config.num_candidates,
        )
        start = time.time()
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            actions = agent(obs=obs)
            action = actions[0]
            action += np.random.normal(
                0,
                np.sqrt(config.action_noise_var),
                env.action_space.shape[0],
            )
            action.clip(min=env.action_space.low, max=env.action_space.high)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            replay_buffer.push(
                observation=obs,
                action=action,
                next_observation=next_obs
            )
            obs = next_obs


        writer.add_scalar('total reward at train', total_reward, episode)
        print('episode [%4d/%4d] is collected. Total reward is %f' %
              (episode+1, config.all_episodes, total_reward))
        print('elasped time for interaction: %.2fs' % (time.time() - start))

        # test without exploration noise
        if (episode + 1) % config.test_interval == 0:
            obs, _ = env.reset()
            done = False
            total_reward = 0
            while not done:
                actions = agent(obs=obs)
                action = actions[0]
                next_obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                obs = next_obs
                done = terminated or truncated

            writer.add_scalar('total reward at test', total_reward, episode)
            print('total test reward at episode [%4d/%4d] is %f' %
                (episode+1, config.all_episodes, total_reward))
            print('elasped time for test: %.2fs' % (time.time() - start))

     # save learned model parameters
    torch.save(encoder.state_dict(), log_dir / "encoder.pth")
    torch.save(decoder.state_dict(), log_dir / "decoder.pth")
    torch.save(transition_model.state_dict(), log_dir / "transition_model.pth")
    writer.close()
    
    return {"model_dir": log_dir}   