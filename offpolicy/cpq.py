from typing import Dict, Iterator
import torch
from tianshou.data import Batch
from fsrl.policy import BasePolicy
from osrl.algorithms import BCQL


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
        This is called by the FastCollector and needs to handle batch processing.
        """
        # The 'obs' from the collector is a numpy array. Convert it to a tensor.
        obs_tensor = torch.tensor(batch.obs, dtype=torch.float32).to(self.device)

        # Ensure the observation tensor is 2D [batch_size, obs_dim]
        if obs_tensor.dim() > 2:
            obs_tensor = obs_tensor.squeeze(1)

        # Directly call the model's actor and VAE for batch processing,
        # bypassing the single-instance `act` method.
        with torch.no_grad():
            action_tensor = self.model.actor(
                obs_tensor, self.model.vae.decode(obs_tensor)
            )

        # Convert the resulting action tensor back to a numpy array for the collector.
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
        )
        offline_batch.to_torch(device=self.device)

        # Trim offline data to the correct size
        offline_batch = offline_batch[:offline_bs]

        # 4. Combine online and offline batches
        mixed_batch = Batch.cat([online_batch, offline_batch])

        # 5. Unpack the mixed batch for training
        # FIX: Unpack the Batch object by directly accessing its attributes,
        # not by using a non-existent `to_tuple` method.
        obs = mixed_batch.obs.float()
        obs_next = mixed_batch.obs_next.float()
        actions = mixed_batch.act.float()
        rewards = mixed_batch.rew.float()
        costs = mixed_batch.cost.float()
        done = mixed_batch.done.float()

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
