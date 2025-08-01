import os
import pprint
import types
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterator, Optional, Tuple, List

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
from fsrl.trainer import OffpolicyTrainer
from fsrl.utils import WandbLogger
from fsrl.utils.exp_util import auto_name, load_config_and_model, seed_all
from osrl.algorithms import COptiDICE
from osrl.common.exp_util import (
    auto_name,
    seed_all,
    DEFAULT_KEY_ABBRE,
    DEFAULT_SKIP_KEY,
)
from osrl.common import TransitionDataset
from offpolicy.coptidice import COptiDICEFinetunePolicy
from config.coptidice_configs import COptiDICEFinetuneConfig


@pyrallis.wrap()
def finetune(args: COptiDICEFinetuneConfig):
    # 1. Load pre-trained model and its original configuration
    original_cfg, model_state = load_config_and_model(args.path)

    # Setup logger
    if args.group is None:
        args.group = original_cfg["task"] + "-cost-" + str(int(args.cost_limit))
    if args.name is None:
        args.name = auto_name(
            asdict(COptiDICEFinetuneConfig()),
            asdict(args),
            args.prefix,
            args.suffix,
            skip_keys=(DEFAULT_SKIP_KEY + ["pretrained_model_path"]),
            key_abbre={
                **DEFAULT_KEY_ABBRE,
                "pretrain_seed": "pretrainseed",
                "finetune_seed": "finetuneseed",
                "trajectory_cost": "trajcost",
            },
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
    # IMPORTANT: Set state_init=True to get the 'is_init' flag for COptiDICE
    offline_dataset = TransitionDataset(
        offline_data,
        reward_scale=original_cfg["reward_scale"],
        cost_scale=original_cfg["cost_scale"],
        state_init=True,
    )

    offline_loader = DataLoader(
        offline_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.threads,
        persistent_workers=True if args.threads > 0 else False,
    )
    offline_loader_iter = iter(offline_loader)

    # Get dataset stats for model initialization
    init_s_propotion, obs_std, act_std = offline_dataset.get_dataset_states()

    # Online data source
    online_buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))

    # 4. Initialize Model
    coptidice_model = COptiDICE(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        f_type=original_cfg["f_type"],
        init_state_propotion=init_s_propotion,
        observations_std=obs_std,
        actions_std=act_std,
        a_hidden_sizes=original_cfg["a_hidden_sizes"],
        c_hidden_sizes=original_cfg["c_hidden_sizes"],
        gamma=original_cfg["gamma"],
        alpha=original_cfg["alpha"],
        cost_ub_epsilon=original_cfg["cost_ub_epsilon"],
        num_nu=original_cfg["num_nu"],
        num_chi=original_cfg["num_chi"],
        cost_limit=args.cost_limit,
        episode_len=original_cfg["episode_len"],
        device=args.device,
    )

    # Load pre-trained weights
    coptidice_model.load_state_dict(model_state["model_state"])
    coptidice_model.to(args.device)

    # IMPORTANT: Initialize new optimizers for fine-tuning
    coptidice_model.setup_optimizers(args.actor_lr, args.critic_lr, args.scalar_lr)

    # 5. Initialize Policy
    policy = COptiDICEFinetunePolicy(
        model=coptidice_model,
        offline_dataloader_iter=offline_loader_iter,
        offline_batch_ratio=args.offline_batch_ratio,
        device=args.device,
        logger=logger,
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
