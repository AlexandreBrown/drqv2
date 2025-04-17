import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from tensordict import TensorDict
from segdac.agents.agent import Agent
from segdac.action_scaling.env_action_scaler import TanhEnvActionScaler
from segdac.agents.action_sampling_strategy import ActionSamplingStrategy
from segdac.data.mdp import MdpData


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class Encoder(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )

        self.apply(weight_init)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class Actor(nn.Module):
    def __init__(
        self, repr_dim: int, action_dim: int, feature_dim: int, hidden_dim: int
    ):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

        self.apply(weight_init)

    def forward(self, obs: torch.Tensor, std: float):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(
        self, repr_dim: int, action_dim: int, feature_dim: int, hidden_dim: int
    ):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class RandomShiftsAug(nn.Module):
    def __init__(self, pad: int):
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class Drqv2ActionSamplingStrategy(ActionSamplingStrategy):
    def __init__(
        self, actor: nn.Module, encoder: nn.Module, stddev_schedule, num_expl_steps: int
    ):
        super().__init__(actor=actor)
        self.encoder = encoder
        self.stddev_schedule = stddev_schedule
        self.scheduler_step = num_expl_steps

    @torch.no_grad()
    def forward(self, mdp_data: MdpData) -> TensorDict:
        b, s, c, h, w = mdp_data.data["pixels"].shape
        obs = self.encoder(mdp_data.data["pixels"].reshape(b, s * c, h, w))
        stddev = schedule(self.stddev_schedule, self.scheduler_step)
        dist = self.actor(obs, stddev)

        if self.is_stochasticity_enabled and self.is_exploration_enabled:
            action = dist.sample(clip=None)
        else:
            action = dist.mean

        return TensorDict(
            {"unscaled_action": action}, batch_size=torch.Size([action.shape[0]])
        )

    def step(self, frames: int = 1):
        self.scheduler_step += frames


class Drqv2Agent(Agent):
    def __init__(
        self,
        env_action_scaler: TanhEnvActionScaler,
        action_sampling_strategy: Drqv2ActionSamplingStrategy,
        action_dim: int,
        device: str,
        feature_dim: int,
        lr: float,
        hidden_dim: int,
        critic_target_tau: float,
        num_expl_steps,
        update_every_steps: int,
        stddev_schedule,
        stddev_clip: float,
        gamma: float,
        num_channels: int,
    ):
        super().__init__(
            env_action_scaler=env_action_scaler,
            action_sampling_strategy=action_sampling_strategy,
        )
        self.critic_target_tau = critic_target_tau
        self.num_expl_steps = num_expl_steps
        self.update_every_steps = update_every_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.gamma = gamma
        self.encoder = Encoder(num_channels=num_channels).to(device)
        self.action_sampling_strategy = self.action_sampling_strategy(
            encoder=self.encoder
        ).to(device)
        self.critic = Critic(
            repr_dim=self.encoder.repr_dim,
            action_dim=action_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
        ).to(device)
        self.critic_target = Critic(
            repr_dim=self.encoder.repr_dim,
            action_dim=action_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    @property
    def actor(self):
        return self.action_sampling_strategy.actor

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        return self

    def update(
        self, train_mdp_data: MdpData, env_step: int, is_time_to_evaluate: bool
    ) -> TensorDict:
        logs_data = {}

        if env_step % self.update_every_steps != 0:
            return TensorDict(
                logs_data,
                batch_size=torch.Size([]),
            )

        self.train()

        b, s, c, h, w = train_mdp_data.data["pixels"].shape
        obs = train_mdp_data.data["pixels"].reshape(b, s * c, h, w)
        action = train_mdp_data.data["action"]
        reward = train_mdp_data.next.data["reward"].reshape(-1, 1)
        done = train_mdp_data.next.data["done"].reshape(-1, 1)
        discount = train_mdp_data.next.data.get(
            "gamma", torch.tensor([self.gamma], device=done.device)
        ).reshape(-1, 1)
        next_obs = train_mdp_data.next.data["pixels"].reshape(b, s * c, h, w)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if is_time_to_evaluate:
            logs_data["batch_reward"] = reward.mean()

        # update critic
        logs_data.update(
            self.update_critic(
                obs,
                action,
                reward,
                discount,
                next_obs,
                env_step,
                done,
                is_time_to_evaluate,
            )
        )

        # update actor
        logs_data.update(self.update_actor(obs.detach(), env_step, is_time_to_evaluate))

        return TensorDict(logs_data, batch_size=torch.Size([]))

    def update_target_networks(self, env_step: int):
        soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

    def update_critic(
        self, obs, action, reward, discount, next_obs, step, done, is_time_to_evaluate
    ):
        metrics = dict()

        with torch.no_grad():
            stddev = schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done.float()) * discount * target_V

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if is_time_to_evaluate:
            metrics["critic_target_q"] = target_Q.mean()
            metrics["critic_q1"] = Q1.detach().mean()
            metrics["critic_q2"] = Q2.detach().mean()
            metrics["critic_loss"] = critic_loss.detach()

        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step, is_time_to_evaluate):
        metrics = dict()

        stddev = schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if is_time_to_evaluate:
            metrics["actor_loss"] = actor_loss.detach()
            metrics["actor_logprob"] = log_prob.detach().mean()
            metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().detach()

        return metrics
