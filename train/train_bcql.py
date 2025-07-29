import os
import pprint
import types
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterator, Tuple, Optional

import bullet_safety_gym  # noqa
import dsrl
import gymnasium as gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
from tianshou.data import Batch, VectorReplayBuffer
from tianshou.env import ShmemVectorEnv
from torch.utils.data import DataLoader

from fsrl.data import FastCollector
from fsrl.policy import BasePolicy
from fsrl.trainer import OffpolicyTrainer
from fsrl.utils import WandbLogger
from fsrl.utils.exp_util import auto_name, load_config_and_model, seed_all
from osrl.algorithms import BCQL
from osrl.common.exp_util import auto_name, seed_all
from osrl.common.net import (
    MLPGaussianPerturbationActor,
    EnsembleDoubleQCritic,
    VAE,
    LagrangianPIDController,
)
from osrl.common import TransitionDataset


@dataclass
class FinetuneConfig:
    # Path to the pre-trained model checkpoint
    path: str = (
        "log/OfflineCarCircle-v0/BCQL-OfflineCarCircle-v0-cost-10-seed-0/checkpoint/model-best.pt"
    )

    # Online training parameters
    epoch: int = 100
    step_per_epoch: int = 1000
    update_per_step: float = 0.5
    buffer_size: int = 200_000
    batch_size: int = 256
    offline_batch_ratio: float = 0.5  # Fraction of the batch from the offline dataset
    episode_per_collect: int = 5

    # Learning rates for fine-tuning (can be smaller than pre-training)
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    vae_lr: float = 1e-4

    # General parameters
    seed: int = 0
    device: str = "cpu"
    threads: int = 4

    # Environment parameters
    cost_limit: float = 10

    # Evaluation parameters
    eval_episodes: int = 10
    eval_every: int = 5  # epochs

    # Logger parameters
    project: str = "OSRL-Finetune"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "BCQL-Finetune"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "logs"
    verbose: bool = True


class BCQLFinetunePolicy(BasePolicy):
    """
    An adapter policy to make osrl.BCQL compatible with fsrl.OffpolicyTrainer.
    This policy implements the mixed-batch update strategy.
    """

    def __init__(
        self,
        model: BCQL,
        offline_dataloader_iter: Iterator,
        offline_batch_ratio: float,
        device: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.offline_dataloader_iter = offline_dataloader_iter
        self.offline_batch_ratio = offline_batch_ratio
        self.device = device

    def forward(self, batch: Batch, state: any = None, **kwargs) -> Batch:
        """
        Given a single obs, return the action, value, logp.
        This is called by the FastCollector.
        """
        obs = batch.obs
        action, _ = self.model.act(obs)
        return Batch(act=action)

    def learn(self, batch: Batch) -> Dict[str, float]:
        """
        The core learning method that performs a mixed-batch update.
        """
        # 1. Determine batch sizes for online and offline data
        total_bs = len(batch.obs)
        offline_bs = int(total_bs * self.offline_batch_ratio)
        online_bs = total_bs - offline_bs

        # 2. Get online data from the replay buffer
        online_batch = batch[:online_bs]

        # 3. Get offline data from the offline dataset iterator
        try:
            offline_data_tuple = next(self.offline_dataloader_iter)
        except StopIteration:
            # If the offline dataloader is exhausted, reset it
            self.offline_dataloader_iter = iter(
                self.offline_dataloader_iter._dataloader
            )
            offline_data_tuple = next(self.offline_dataloader_iter)

        # Convert the tuple from DataLoader to a Tianshou/FSRL Batch object
        offline_batch = Batch(
            obs=offline_data_tuple[0],
            obs_next=offline_data_tuple[1],
            act=offline_data_tuple[2],
            rew=offline_data_tuple[3],
            cost=offline_data_tuple[4],
            done=offline_data_tuple[5],
        ).to(self.device)
        # Trim offline data to the correct size
        offline_batch = offline_batch[:offline_bs]

        # 4. Combine online and offline batches
        mixed_batch = Batch.cat([online_batch, offline_batch])
        mixed_batch.to_torch(device=self.device)

        # 5. Unpack the mixed batch for training
        obs, obs_next, actions, rewards, costs, done = [
            b.float() for b in mixed_batch.to_tuple()
        ]

        # 6. Perform the updates using the mixed batch
        all_stats = {}
        _, stats_vae = self.model.vae_loss(obs, actions)
        _, stats_critic = self.model.critic_loss(obs, obs_next, actions, rewards, done)
        _, stats_cost_critic = self.model.cost_critic_loss(
            obs, obs_next, actions, costs, done
        )
        _, stats_actor = self.model.actor_loss(obs)
        self.model.sync_weight()

        all_stats.update(stats_vae)
        all_stats.update(stats_critic)
        all_stats.update(stats_cost_critic)
        all_stats.update(stats_actor)

        return all_stats

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


@pyrallis.wrap()
def finetune(args: FinetuneConfig):
    # 1. Load pre-trained model and its original configuration
    original_cfg, model_state = load_config_and_model(args.path)

    # Setup logger
    if args.group is None:
        args.group = original_cfg["task"] + "-cost-" + str(int(args.cost_limit))
    if args.name is None:
        args.name = auto_name(
            asdict(args), asdict(FinetuneConfig()), args.prefix, args.suffix
        )
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.project, args.group, args.name)

    logger = WandbLogger(asdict(args), args.project, args.group, args.name, args.logdir)
    logger.save_config(asdict(args), verbose=args.verbose)

    # Set seed
    seed_all(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    # 2. Initialize Environment
    env = gym.make(original_cfg["task"])
    env.set_target_cost(args.cost_limit)

    train_envs = ShmemVectorEnv(
        [
            lambda: gym.make(original_cfg["task"])
            for _ in range(args.episode_per_collect)
        ]
    )
    test_envs = ShmemVectorEnv(
        [lambda: gym.make(original_cfg["task"]) for _ in range(args.eval_episodes)]
    )

    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # 3. Prepare Data Sources
    # Offline data source
    offline_data = env.get_dataset()
    offline_dataset = TransitionDataset(
        offline_data,
        reward_scale=original_cfg["reward_scale"],
        cost_scale=original_cfg["cost_scale"],
    )

    # Use a persistent worker to avoid re-initialization overhead
    offline_loader = DataLoader(
        offline_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.threads,
        persistent_workers=True,
    )
    offline_loader_iter = iter(offline_loader)

    # Online data source
    online_buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))

    # 4. Initialize Model
    bcql_model = BCQL(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        a_hidden_sizes=original_cfg["a_hidden_sizes"],
        c_hidden_sizes=original_cfg["c_hidden_sizes"],
        vae_hidden_sizes=original_cfg["vae_hidden_sizes"],
        sample_action_num=original_cfg["sample_action_num"],
        PID=original_cfg["PID"],
        gamma=original_cfg["gamma"],
        tau=original_cfg["tau"],
        lmbda=original_cfg["lmbda"],
        beta=original_cfg["beta"],
        phi=original_cfg["phi"],
        num_q=original_cfg["num_q"],
        num_qc=original_cfg["num_qc"],
        cost_limit=args.cost_limit,
        episode_len=original_cfg["episode_len"],
        device=args.device,
    )

    # Load pre-trained weights
    bcql_model.load_state_dict(model_state["model_state"])
    bcql_model.to(args.device)

    # IMPORTANT: Initialize new optimizers for fine-tuning
    bcql_model.setup_optimizers(args.actor_lr, args.critic_lr, args.vae_lr)

    # 5. Initialize Policy
    policy = BCQLFinetunePolicy(
        model=bcql_model,
        offline_dataloader_iter=offline_loader_iter,
        offline_batch_ratio=args.offline_batch_ratio,
        device=args.device,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    # 6. Initialize Collectors and Trainer
    train_collector = FastCollector(
        policy,
        train_envs,
        online_buffer,
        exploration_noise=True,
    )
    test_collector = FastCollector(policy, test_envs)

    def checkpoint_fn():
        return {"model": policy.state_dict()}

    logger.setup_checkpoint_fn(checkpoint_fn)

    trainer = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        batch_size=args.batch_size,
        cost_limit=args.cost_limit,
        step_per_epoch=args.step_per_epoch,
        update_per_step=args.update_per_step,
        episode_per_test=args.eval_episodes,
        episode_per_collect=args.episode_per_collect,
        logger=logger,
        verbose=args.verbose,
    )

    # 7. Run Fine-tuning
    for epoch, epoch_stat, info in trainer:
        logger.store(tab="train", cost_limit=args.cost_limit)
        print(f"Epoch: {epoch}")
        print(info)

    if __name__ == "__main__":
        pprint.pprint(info)
        # Final evaluation
        env = gym.make(original_cfg["task"])
        policy.eval()
        collector = FastCollector(policy, env)
        result = collector.collect(n_episode=10, render=False)
        rews, lens, cost = result["rew"], result["len"], result["cost"]
        print(
            f"Final eval reward: {rews.mean()}, cost: {cost.mean()}, length: {lens.mean()}"
        )


if __name__ == "__main__":
    finetune()
