from dataclasses import dataclass
from typing import Optional

@dataclass
class CPQFinetuneConfig:
    # Path to the pre-trained model checkpoint
    path: str = (
        "log/OfflineCarCircle-v0/CPQ-OfflineCarCircle-v0-cost-10-seed-0/checkpoint/model-best.pt"
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
    actor_lr: float = 1e-5
    critic_lr: float = 1e-4
    alpha_lr: float = 1e-5
    vae_lr: float = 1e-4

    # General parameters
    pretrain_seed: int = 0
    finetune_seed: int = 0
    device: str = "cuda"
    threads: int = 8

    # Environment parameters
    cost_limit: float = 0

    # Evaluation parameters
    eval_episodes: int = 10
    eval_every: int = 5  # epochs

    # Logger parameters
    project: str = "OSRL-Finetune"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "CPQ-Finetune"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "logs-finetune"
    verbose: bool = True
