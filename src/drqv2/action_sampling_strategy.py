import torch
import torch.nn as nn
import drqv2.utils as utils
from segdac.agents.action_sampling_strategy import ActionSamplingStrategy


class DrqV2ActionSamplingStrategy(ActionSamplingStrategy):
    def __init__(self, actor: nn.Module, encoder: nn.Module, stddev_schedule):
        super().__init__(actor=actor)
        self.encoder = encoder
        self.stddev_schedule = stddev_schedule
        self.env_step = 1

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        obs = self.encoder(x)
        stddev = utils.schedule(self.stddev_schedule, self.env_step)
        dist = self.actor(obs, stddev)

        if self.stochasticity_enabled:
            action = dist.sample(clip=None)
            self.env_step += 1
        else:
            action = dist.mean

        return action
