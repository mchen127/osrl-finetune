from typing import Dict, Iterator
import torch
from tianshou.data import Batch
from fsrl.policy import BasePolicy
from osrl.algorithms import CPQ


class CPQFinetunePolicy(BasePolicy):
    """
    An adapter policy to make osrl.CPQ compatible with fsrl.OffpolicyTrainer.
    This policy implements the mixed-batch update strategy.
    """

    def __init__(
        self,
        model: CPQ,
        offline_dataloader_iter: Iterator,
        offline_batch_ratio: float,
        device: str,
        **kwargs,
    ):
        super().__init__(
            actor=model.actor, critics=[model.critic, model.cost_critic], **kwargs
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
            # Use self.training to toggle deterministic actions for exploration vs evaluation
            is_deterministic = not self.training
            action_tensor, _ = self.model.act(
                obs_tensor, deterministic=is_deterministic
            )

        action = action_tensor.cpu().numpy()
        return Batch(act=action)

    def learn(self, batch: Batch) -> Dict[str, float]:
        """
        The core learning method that performs a mixed-batch update.
        """
        # 1. Determine batch sizes
        total_bs = len(batch.obs)
        offline_bs = int(total_bs * self.offline_batch_ratio)
        online_bs = total_bs - offline_bs

        # 2. Get online data and move to device
        online_batch = batch[:online_bs]
        online_batch.to_torch(device=self.device)

        # 3. Get offline data and move to device
        try:
            offline_data_tuple = next(self.offline_dataloader_iter)
        except StopIteration:
            self.offline_dataloader_iter = iter(
                self.offline_dataloader_iter._dataloader
            )
            offline_data_tuple = next(self.offline_dataloader_iter)

        offline_batch = Batch(
            {
                k: v
                for k, v in zip(
                    ["obs", "obs_next", "act", "rew", "cost", "done"],
                    offline_data_tuple,
                )
            }
        )
        offline_batch.to_torch(device=self.device)
        offline_batch = offline_batch[:offline_bs]

        # 4. Combine batches
        mixed_batch = Batch.cat([online_batch, offline_batch])

        # 5. Unpack for training
        obs = mixed_batch.obs.float()
        obs_next = mixed_batch.obs_next.float()
        actions = mixed_batch.act.float()
        rewards = mixed_batch.rew.float()
        costs = mixed_batch.cost.float()
        done = mixed_batch.done.float()

        # 6. Perform sequential updates
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
        
        self.logger.store(**stats_vae)
        self.logger.store(**stats_critic)
        self.logger.store(**stats_cost_critic)
        self.logger.store(**stats_actor)

        return all_stats

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
