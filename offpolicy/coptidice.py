from typing import Dict, Iterator
import torch
from tianshou.data import Batch
from osrl.algorithms import COptiDICE
from fsrl.policy import BasePolicy


class COptiDICEFinetunePolicy(BasePolicy):
    """
    An adapter policy to make osrl.COptiDICE compatible with fsrl.OffpolicyTrainer.
    This policy implements the mixed-batch update strategy.
    """

    def __init__(
        self,
        model: COptiDICE,
        offline_dataloader_iter: Iterator,
        offline_batch_ratio: float,
        device: str,
        **kwargs,
    ):
        super().__init__(
            actor=model.actor, critics=[model.nu_network, model.chi_network], **kwargs
        )
        self.model = model
        self.offline_dataloader_iter = offline_dataloader_iter
        self.offline_batch_ratio = offline_batch_ratio
        self.device = device

    def forward(self, batch: Batch, state: any = None, **kwargs) -> Batch:
        """
        Get actions for a batch of observations.
        """
        obs_tensor = torch.tensor(batch.obs, dtype=torch.float32).to(self.device)
        if obs_tensor.dim() > 2:
            obs_tensor = obs_tensor.squeeze(1)

        with torch.no_grad():
            # The trainer calls policy.train() or policy.eval() to set self.training.
            # We use this flag to decide whether to use deterministic actions.
            # self.training = True for exploration (stochastic), False for evaluation (deterministic).
            is_deterministic = not self.training
            action_tensor, _ = self.model.actor.forward(
                obs_tensor, deterministic=is_deterministic
            )

        action = action_tensor.cpu().numpy()
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
        online_batch.to_torch(device=self.device)

        # 3. Get offline data from the offline dataset iterator
        try:
            offline_data_tuple = next(self.offline_dataloader_iter)
        except StopIteration:
            self.offline_dataloader_iter = iter(
                self.offline_dataloader_iter._dataloader
            )
            offline_data_tuple = next(self.offline_dataloader_iter)

        # Unpack offline data and move to device
        obs_off, obs_next_off, act_off, rew_off, cost_off, done_off, is_init_off = [
            d.to(self.device) for d in offline_data_tuple
        ]

        # Trim offline data to the correct size
        obs_off, obs_next_off, act_off, rew_off, cost_off, done_off, is_init_off = (
            obs_off[:offline_bs],
            obs_next_off[:offline_bs],
            act_off[:offline_bs],
            rew_off[:offline_bs],
            cost_off[:offline_bs],
            done_off[:offline_bs],
            is_init_off[:offline_bs],
        )

        # 4. Combine online and offline batches
        # Online data is never the initial state of a trajectory in the offline sense
        is_init_on = torch.zeros_like(online_batch.done).to(self.device)

        obs = torch.cat([online_batch.obs, obs_off], dim=0)
        obs_next = torch.cat([online_batch.obs_next, obs_next_off], dim=0)
        actions = torch.cat([online_batch.act, act_off], dim=0)
        rewards = torch.cat([online_batch.rew, rew_off], dim=0)
        costs = torch.cat([online_batch.cost, cost_off], dim=0)
        done = torch.cat([online_batch.done, done_off], dim=0)
        is_init = torch.cat([is_init_on, is_init_off], dim=0)

        # 5. Perform the update using the mixed batch
        mixed_batch_tuple = (
            obs.float(),
            obs_next.float(),
            actions.float(),
            rewards.float(),
            costs.float(),
            done.float(),
            is_init.float(),
        )

        stats_loss = self.model.update(mixed_batch_tuple)
        self.logger.store(**stats_loss)

        return stats_loss

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
