import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

class WarmupCosineScheduler(LRScheduler):

    def __init__(self,
                optimizer: Optimizer,
                num_warmup_steps: int,
                num_training_steps: int,
                num_cycles: float = 0.5,
                ):

        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        if self.current_step < self.num_warmup_steps:
            return (self.current_step / max(1,self.num_warmup_steps)) * self.optimizer.defaults['lr']
        progress = (self.current_step - self.num_warmup_steps) / max(1, self.num_training_steps - self.num_warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * self.num_cycles * progress)) * self.optimizer.defaults['lr']

    def state_dict(self):

        return {
        'num_warmup_steps': self.num_warmup_steps,
        'num_training_steps': self.num_training_steps,
        'num_cycles': self.num_cycles,
        'current_step': self.current_step 
        }

    def load_state_dict(self, state_dict):
        self.num_warmup_steps = state_dict['num_warmup_steps']
        self.num_training_steps = state_dict['num_training_steps']
        self.num_cycles = state_dict['num_cycles']
        self.current_step = state_dict['current_step']