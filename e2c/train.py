import os
import json
import torch
import numpy as np
import gymnasium as gym
from pathlib import Path
from datetime import datetime
from torch.distributions.kl import kl_divergence
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from .memory import StaticDataset
from .configs import TrainConfig
from .models import (
    Encoder,
    Decoder,
    TransitionModel,
)
from tqdm import tqdm
from torch.distributions import MultivariateNormal


def collect_data(env: gym.Env, num_episodes: int):
    dataset = StaticDataset(
        capacity=num_episodes*env.horizon,
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
    )

    print("collecting data")
    for _ in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            dataset.push(
                observation=obs,
                action=action,
                next_observation=next_obs
            )
            obs = next_obs

    return dataset


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

    # create datasets
    dataset = collect_data(env=env, num_episodes=config.num_episodes)
    train_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=[1-config.test_size, config.test_size],
    )

    # create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
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

    for epoch in range(config.num_epochs):
        # train
        encoder.train()
        decoder.train()
        transition_model.train()
        for batch, observations, actions, next_observations in enumerate(train_dataloader):

            observations = observations.to(device)
            actions = actions.to(device)
            next_observations = next_observations.to(device)

            priors = MultivariateNormal(
                torch.zeros((config.batch_size, config.state_dim), device=device, dtype=torch.float32),
                torch.diag_embed(torch.ones((config.batch_size, config.state_dim), device=device, dtype=torch.float32)),
            )
            posteriors = encoder(observations)
            posterior_samples = posteriors.rsample()
            next_priors, _, _, _ = transition_model(
                state_sample=posterior_samples,
                action=actions,
                state_dist=posteriors
            )
            next_prior_samples = next_priors.rsample()
            next_posteriors = encoder(next_observations)

            kl = kl_divergence(posteriors, priors)
            kl_loss = kl.clamp(min=config.free_nats).mean()

            next_kl = kl_divergence(next_priors, next_posteriors)
            next_kl_loss = next_kl.clamp(min=config.free_nats).mean()

            recon_observations = decoder(posterior_samples)
            recon_next_observations = decoder(next_prior_samples)

            obs_loss = mse_loss(
                recon_observations,
                observations,
            )

            next_obs_loss = mse_loss(
                recon_next_observations,
                next_observations,
            )

            loss = obs_loss + next_obs_loss + config.kl_beta * next_kl_loss + kl_loss
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(all_params, config.clip_grad_norm)
            optimizer.step()

            total_idx = epoch * len(train_dataloader) + batch 
        
            writer.add_scalar('overall loss train', loss.item(), total_idx)
            writer.add_scalar('kl loss train', kl_loss.item(), total_idx)
            writer.add_scalar('next_kl loss train', next_kl_loss.item(), total_idx)
            writer.add_scalar('obs loss train', obs_loss.item(), total_idx)
            writer.add_scalar('next obs loss train', next_obs_loss.item(), total_idx)

        # test
        encoder.eval()
        decoder.eval()
        transition_model.eval()

        with torch.no_grad():

            for batch, observations, actions, next_observations in enumerate(test_dataloader):

                observations = observations.to(device)
                actions = actions.to(device)
                next_observations = next_observations.to(device)

                priors = MultivariateNormal(
                    torch.zeros((config.batch_size, config.state_dim), device=device, dtype=torch.float32),
                    torch.diag_embed(torch.ones((config.batch_size, config.state_dim), device=device, dtype=torch.float32)),
                )
                posteriors = encoder(observations)
                posterior_samples = posteriors.sample()
                next_priors, _, _, _ = transition_model(
                    state_sample=posterior_samples,
                    action=actions,
                    state_dist=posteriors
                )
                next_prior_samples = next_priors.sample()
                next_posteriors = encoder(next_observations)

                kl = kl_divergence(posteriors, priors)
                kl_loss = kl.clamp(min=config.free_nats).mean()

                next_kl = kl_divergence(next_priors, next_posteriors)
                next_kl_loss = next_kl.clamp(min=config.free_nats).mean()

                recon_observations = decoder(posterior_samples)
                recon_next_observations = decoder(next_prior_samples)

                obs_loss = mse_loss(
                    recon_observations,
                    observations,
                )

                next_obs_loss = mse_loss(
                    recon_next_observations,
                    next_observations,
                )

                total_idx = epoch * len(test_dataloader) + batch 
            
                writer.add_scalar('overall loss test', loss.item(), total_idx)
                writer.add_scalar('kl loss test', kl_loss.item(), total_idx)
                writer.add_scalar('next_kl loss test', next_kl_loss.item(), total_idx)
                writer.add_scalar('obs loss test', obs_loss.item(), total_idx)
                writer.add_scalar('next obs loss test', next_obs_loss.item(), total_idx)

     # save learned model parameters
    torch.save(encoder.state_dict(), log_dir / "encoder.pth")
    torch.save(decoder.state_dict(), log_dir / "decoder.pth")
    torch.save(transition_model.state_dict(), log_dir / "transition_model.pth")
    writer.close()
    
    return {"model_dir": log_dir}   