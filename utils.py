"""
Utility functions for the lie detection system.
"""
import random
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


def set_seed(seed: int, config: dict = None) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
        config: Configuration dictionary
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if config is not None:
        if "cudnn_deterministic_toggle" in config:
            torch.backends.cudnn.deterministic = str_to_bool(config["cudnn_deterministic_toggle"])
        else:
            torch.backends.cudnn.deterministic = True
            
        if "cudnn_benchmark_toggle" in config:
            torch.backends.cudnn.benchmark = str_to_bool(config["cudnn_benchmark_toggle"])
        else:
            torch.backends.cudnn.benchmark = False
            
        config["seed"] = seed


def seed_worker(worker_id: int) -> None:
    """
    Set random seed for DataLoader workers.
    
    Args:
        worker_id: Worker ID
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def str_to_bool(val: str) -> bool:
    """
    Convert string to boolean.
    
    Args:
        val: String value
        
    Returns:
        Boolean value
    """
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError("Boolean value expected.")


def create_optimizer(params, config: dict):
    """
    Create optimizer and learning rate scheduler.
    
    Args:
        params: Model parameters
        config: Optimizer configuration
        
    Returns:
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
    """
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            amsgrad=config["amsgrad"]
        )
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config["lr"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
            nesterov=config["nesterov"]
        )
    else:
        raise ValueError("Optimizer not supported: {}".format(config["optimizer"]))
    
    # Learning rate scheduler
    if config["scheduler"] == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config["epochs"] * config["steps_per_epoch"],
            eta_min=config["min_lr"]
        )
    elif config["scheduler"] == "keras_decay":
        scheduler = KerasDecay(
            optimizer,
            steps_per_epoch=config["steps_per_epoch"],
            initial_epoch=0,
            **config["keras_decay_params"]
        )
    else:
        scheduler = None
    
    return optimizer, scheduler


class KerasDecay:
    """
    Learning rate scheduler that mimics Keras learning rate decay.
    """
    def __init__(self, optimizer, steps_per_epoch, initial_epoch=0, decay=0.1, decay_epochs=30):
        """
        Args:
            optimizer: PyTorch optimizer
            steps_per_epoch: Number of steps per epoch
            initial_epoch: Initial epoch
            decay: Decay factor
            decay_epochs: Number of epochs after which to decay the learning rate
        """
        self.optimizer = optimizer
        self.steps_per_epoch = steps_per_epoch
        self.epoch = initial_epoch
        self.decay = decay
        self.decay_epochs = decay_epochs
        self.initial_lr = optimizer.param_groups[0]['lr']
        
    def step(self):
        """Update learning rate"""
        step_num = self.epoch * self.steps_per_epoch
        decay_steps = self.decay_epochs * self.steps_per_epoch
        
        if step_num % decay_steps == 0 and step_num > 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * (1.0 - self.decay)
        
        self.epoch += 1 / self.steps_per_epoch 