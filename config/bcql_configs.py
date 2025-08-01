from dataclasses import dataclass
from typing import Optional


@dataclass
class BCQLFinetuneConfig:
    # Path to the pre-trained model checkpoint
    pretrained_model_path: str = (
        "logs-osrl/OfflineHalfCheetahVelocityGymnasium-v1-cost-20/BCQL_cost20_seed10-0df4/BCQL_cost20_seed10-0df4"
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
    pretrain_seed: int = 0
    finetune_seed: int = 0
    device: str = "cuda"
    threads: int = 8

    # Environment parameters
    cost_limit: int = 0
    trajectory_cost: str = "all"

    # Evaluation parameters
    eval_episodes: int = 10
    eval_every: int = 5  # epochs

    # Logger parameters
    project: str = "OSRL-Finetune"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "BCQL-Finetune"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "logs-finetune"
    verbose: bool = True
