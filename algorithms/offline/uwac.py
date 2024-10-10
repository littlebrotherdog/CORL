# Inspired by:
# 1. paper for UWAC: https://arxiv.org/abs/2105.08140
# 2. implementation: https://github.com/apple/ml-uwac
import math
import os
import abc
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb
from torch.distributions import Distribution, Normal
from tqdm import trange

@dataclass
class TrainConfig:
    # wandb params
    project: str = "CORL"
    group: str = "UWAC"
    name: str = "UWAC"
    # model params
    hidden_dim: int = 256
    num_critics: int = 2
    gamma: float = 0.99
    tau: float = 5e-3
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    max_action: float = 1.0
    # training params
    buffer_size: int = 2_000_000
    env_name: str = "halfcheetah-medium-v2"
    batch_size: int = 256
    num_epochs: int = 3000
    num_updates_on_epoch: int = 1000
    normalize_reward: bool = False
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 5
    # general params
    checkpoints_path: Optional[str] = None
    deterministic_torch: bool = False
    train_seed: int = 10
    eval_seed: int = 42
    log_every: int = 100
    device: str = "cuda"
    # uwac params
    mmd_sigma = 20
    kernel_type = 'gaussian'
    target_mmd_thresh = 0.07
    num_samples = 100
    drop_rate = 0.1
    beta = 0.5
    clip_bottom = 0.0
    clip_top = 1.0
    use_exp_weight = True
    var_Pi = True
    use_exp_penalty = False
    SN = True

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
MEAN_MIN = -9.0
MEAN_MAX = 9.0

def atanh(x):
    one_plus_x = (1 + x).clamp(min=1e-6)
    one_minus_x = (1 - x).clamp(min=1e-6)
    return 0.5*torch.log(one_plus_x/ one_minus_x)

# general utils
TensorBatch = List[torch.Tensor]


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env

device = "cuda"

def identity(x):
    return x
def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)
def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)
def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
def zeros(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros(*sizes, **kwargs, device=torch_device)
def ones(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones(*sizes, **kwargs, device=torch_device)
def torch_ify(np_array_or_other):
    if isinstance(np_array_or_other, np.ndarray):
        return from_numpy(np_array_or_other)
    else:
        return np_array_or_other
def np_ify(tensor_or_other):
    if isinstance(tensor_or_other, torch.autograd.Variable):
        return get_numpy(tensor_or_other)
    else:
        return tensor_or_other
def eval_np(module, *args, **kwargs):
    """
    Eval this module with a numpy interface

    Same as a call to __call__ except all Variable input/outputs are
    replaced with numpy equivalents.

    Assumes the output is either a single object or a tuple of objects.
    """
    torch_args = tuple(torch_ify(x) for x in args)
    torch_kwargs = {k: torch_ify(v) for k, v in kwargs.items()}
    outputs = module(*torch_args, **torch_kwargs)
    if isinstance(outputs, tuple):
        return tuple(np_ify(x) for x in outputs)
    else:
        return np_ify(outputs)

class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cuda",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        raise NotImplementedError

class FixedNormalizer(object):
    def __init__(
            self,
            size,
            default_clip_range=np.inf,
            mean=0,
            std=1,
            eps=1e-8,
    ):
        assert std > 0
        std = std + eps
        self.size = size
        self.default_clip_range = default_clip_range
        self.mean = mean + np.zeros(self.size, np.float32)
        self.std = std + np.zeros(self.size, np.float32)
        self.eps = eps

    def set_mean(self, mean):
        self.mean = mean + np.zeros(self.size, np.float32)

    def set_std(self, std):
        std = std + self.eps
        self.std = std + np.zeros(self.size, np.float32)

    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        mean, std = self.mean, self.std
        if v.ndim == 2:
            mean = mean.reshape(1, -1)
            std = std.reshape(1, -1)
        return np.clip((v - mean) / std, -clip_range, clip_range)

    def denormalize(self, v):
        mean, std = self.mean, self.std
        if v.ndim == 2:
            mean = mean.reshape(1, -1)
            std = std.reshape(1, -1)
        return mean + v * std

    def copy_stats(self, other):
        self.set_mean(other.mean)
        self.set_std(other.std)

class TorchFixedNormalizer(FixedNormalizer):
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        mean = ptu.from_numpy(self.mean)
        std = ptu.from_numpy(self.std)
        if v.dim() == 2:
            # Unsqueeze along the batch use automatic broadcasting
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return torch.clamp((v - mean) / std, -clip_range, clip_range)

    def normalize_scale(self, v):
        """
        Only normalize the scale. Do not subtract the mean.
        """
        std = ptu.from_numpy(self.std)
        if v.dim() == 2:
            std = std.unsqueeze(0)
        return v / std

    def denormalize(self, v):
        mean = ptu.from_numpy(self.mean)
        std = ptu.from_numpy(self.std)
        if v.dim() == 2:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return mean + v * std

    def denormalize_scale(self, v):
        """
        Only denormalize the scale. Do not add the mean.
        """
        std = ptu.from_numpy(self.std)
        if v.dim() == 2:
            std = std.unsqueeze(0)
        return v * std

class Policy(object, metaclass=abc.ABCMeta):
    """
    General policy interface.
    """
    @abc.abstractmethod
    def get_action(self, observation):
        """

        :param observation:
        :return: action, debug_dictionary
        """
        pass

    def reset(self):
        pass

class ExplorationPolicy(Policy, metaclass=abc.ABCMeta):
    def set_num_steps_total(self, t):
        pass

class Dropout_Mlp(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
            drop_rate=0.1,
            spectral_norm=False,
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()
        self.device = 'cuda'
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size).to(self.device)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            if spectral_norm:
                fc = nn.utils.spectral_norm(fc)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        last_fc = nn.Linear(in_size, output_size).to(self.device)
        last_fc.weight.data.uniform_(-init_w, init_w).to(self.device)
        last_fc.bias.data.uniform_(-init_w, init_w).to(self.device)
        if spectral_norm:
            self.last_fc = nn.utils.spectral_norm(last_fc)
        else:
            self.last_fc = last_fc
        self.drop_rate = drop_rate

    def forward(self, input, return_preactivations=False):
        h = input.to(self.device)
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
            h = F.dropout(h, p=self.drop_rate)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

class FlattenDropout_Mlp(Dropout_Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def sample(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)

    def multiple(self, *inputs, num_samples=100, with_var=False, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        shape = list(flat_inputs.size())
        flat_inputs = flat_inputs.unsqueeze(1).expand([shape[0], num_samples] + shape[1:]).reshape(-1, *shape[
                                                                                                        1:]).contiguous()
        output = super().forward(flat_inputs, **kwargs)
        output = output.view(shape[0], num_samples, *output.shape[1:])
        if with_var:
            return output.mean(1), torch.var(output, dim=1)
        return output.mean(1)

    def forward(self, *inputs, **kwargs):
        return self.multiple(*inputs, **kwargs)

class Mlp(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.device = torch.device("cuda")
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size).to(self.device)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size).to(self.device)
        self.last_fc.weight.data.uniform_(-init_w, init_w).to(self.device)
        self.last_fc.bias.data.uniform_(-init_w, init_w).to(self.device)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

class FlattenMlp(Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)

class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return eval_np(self, obs)

class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, output_activation=torch.tanh, **kwargs)

class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1+value) / (1-value)
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        z = (
            self.normal_mean +
            self.normal_std *
            Normal(
                zeros(self.normal_mean.size()),
                ones(self.normal_std.size())
            ).sample()
        )
        z.requires_grad_()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

class TanhGaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            max_action: float = 1.0,
            device=None,  # 添加设备参数
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.device = device if device is not None else torch.device("cuda")
        self.log_std = None
        self.std = std
        self.max_action = max_action
        self.action_dim = action_dim
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim).to(self.device)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w).to(self.device)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w).to(self.device)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def log_prob(self, obs, actions):
        raw_actions = atanh(actions)
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        mean = torch.clamp(mean, MEAN_MIN, MEAN_MAX)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(value=actions, pre_tanh_value=raw_actions)
        return log_prob.sum(-1)

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        # 将输入的 obs 移动到模型的设备上
        obs = obs.to(self.device)

        h = obs
        h = h.to(self.device)
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h)).to(self.device)

        mean = self.last_fc(h).to(self.device)

        if self.std is None:
            log_std = self.last_fc_log_std(h).to(self.device)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std).to(self.device)
        else:
            std = self.std.to(self.device)
            log_std = self.log_std.to(self.device)

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None

        if deterministic:
            action = torch.tanh(mean).to(self.device)
        else:
            tanh_normal = TanhNormal(mean, std)

            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )

                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                ).to(self.device)
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample().to(self.device)
                else:
                    action = tanh_normal.sample().to(self.device)

        return (
            action.to(self.device), mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )

class VAEPolicy(Mlp, ExplorationPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            latent_dim,
            std=None,
            init_w=1e-3,
            device=None,  # 添加设备参数
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.device = device if device is not None else torch.device("cuda")
        self.latent_dim = latent_dim

        # 定义网络层，并移动到指定设备
        self.e1 = torch.nn.Linear(obs_dim + action_dim, 750).to(self.device)
        self.e2 = torch.nn.Linear(750, 750).to(self.device)

        self.mean = torch.nn.Linear(750, self.latent_dim).to(self.device)
        self.log_std = torch.nn.Linear(750, self.latent_dim).to(self.device)

        self.d1 = torch.nn.Linear(obs_dim + self.latent_dim, 750).to(self.device)
        self.d2 = torch.nn.Linear(750, 750).to(self.device)
        self.d3 = torch.nn.Linear(750, action_dim).to(self.device)

        self.max_action = 1.0
        self.latent_dim = latent_dim

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic, execute_actions=True)[0]

    def forward(self, state, action):
        # 确保 state 和 action 在同一个设备上
        state = state.to(self.device)
        action = action.to(self.device)

        # 前向传播
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)

        # 随机采样 eps 并确保其设备与 std 一致
        eps = torch.randn_like(std)
        z = mean + std * eps

        # 解码
        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None):
        state = state.to(self.device)
        if z is None:
            # 确保 numpy 生成的数据转换为 PyTorch 张量，并移动到正确设备
            z = torch.from_numpy(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).float().to(self.device)
            z = z.clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return torch.tanh(self.d3(a))

    def decode_multiple(self, state, z=None, num_decode=10):
        state = state.to(self.device)
        if z is None:
            # 确保 numpy 生成的数据转换为 PyTorch 张量，并移动到正确设备
            z = torch.from_numpy(np.random.normal(0, 1, size=(state.size(0), num_decode, self.latent_dim))).float().to(self.device)
            z = z.clamp(-0.5, 0.5)

        # 确保 state 和 z 维度一致，且张量在同一设备上
        state_repeated = state.unsqueeze(0).repeat(num_decode, 1, 1).permute(1, 0, 2).to(self.device)
        a = F.relu(self.d1(torch.cat([state_repeated, z], 2)))
        a = F.relu(self.d2(a))
        return torch.tanh(self.d3(a)), self.d3(a)



class UWAC:
    def __init__(
        self,
        policy,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        vae,
        policy_optimizer,
        vae_policy_optimizer,
        qf1_optimizer,
        qf2_optimizer,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_learning_rate: float = 1e-4,
        device: str = "cuda",
        # uwac params
        mode = 'auto',
        kernel_choice='laplacian',
        policy_update_style = '0',
        mmd_sigma=10.0,
        target_mmd_thresh=0.05,
        num_samples_mmd_match=4,
        # Dropout specific params
        beta=0.5,
        clip_bottom=0.0,
        clip_top=1.0,
        use_exp_weight=True,
        var_Pi=False,
        q_penalty=0.0,
        use_exp_penalty=False,
        reward_scale = 1.0,
        SN = True,
    ):
        self.device = device
        self.policy = policy
        self.vae = vae
        self.qf1 = qf1
        self.qf2 = qf2

        self.policy_optimizer = policy_optimizer
        self.vae_optimizer = vae_policy_optimizer
        self.qf1_optimizer=qf1_optimizer
        self.qf2_optimizer=qf2_optimizer
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2

        self.tau = tau
        self.gamma = gamma


        # uwac params
        self.mode = mode
        if self.mode == 'auto':
            self.log_alpha = zeros(1, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha],lr=1e-3,)
        self.mmd_sigma = mmd_sigma
        self.kernel_choice = kernel_choice
        self.num_samples_mmd_match = num_samples_mmd_match
        self.policy_update_style = policy_update_style
        self.target_mmd_thresh = target_mmd_thresh
        self.discount = gamma
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.discrete = False
        self._current_epoch = 0
        self._policy_update_ctr = 0
        self._num_q_update_steps = 0
        self._num_policy_update_steps = 0

        self.beta = beta
        self.var_Pi = var_Pi
        self.use_exp_weight = use_exp_weight
        self.clip_top = clip_top
        self.clip_bottom = clip_bottom
        self.q_penalty = q_penalty
        self.reward_scale = reward_scale
        self.use_exp_penalty = use_exp_penalty

    def mmd_loss_laplacian(self, samples1, samples2, sigma=0.2):
        """MMD constraint with Laplacian kernel for support matching"""
        # sigma is set to 20.0 for hopper, cheetah and 50 for walker/ant
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
        diff_x_x = torch.mean((-(diff_x_x.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        diff_x_y = torch.mean((-(diff_x_y.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d
        diff_y_y = torch.mean((-(diff_y_y.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss

    def mmd_loss_gaussian(self, samples1, samples2, sigma=0.2):
        """MMD constraint with Gaussian Kernel support matching"""
        # sigma is set to 20.0 for hopper, cheetah and 50 for walker/ant
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
        diff_x_x = torch.mean((-(diff_x_x.pow(2)).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        diff_x_y = torch.mean((-(diff_x_y.pow(2)).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d
        diff_y_y = torch.mean((-(diff_y_y.pow(2)).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss

    def _get_weight(self, var, LAMBDA, factor=1):
        if self.use_exp_weight:
            weight = torch.clamp(torch.exp(-LAMBDA * var / factor), self.clip_bottom, self.clip_top)
        else:
            weight = torch.clamp(LAMBDA * factor / var, self.clip_bottom, self.clip_top)
        return weight

    def _actor_loss(self, state: torch.Tensor,actions: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        sampled_actions, raw_sampled_actions = self.vae.decode_multiple(state, num_decode=self.num_samples_mmd_match)
        actor_samples, _, _, _, batch_entropy, _, _, raw_actor_actions = self.policy(
            state.unsqueeze(1).repeat(1, self.num_samples_mmd_match, 1).view(-1, state.shape[1]), return_log_prob=True)
        actor_samples = actor_samples.view(state.shape[0], self.num_samples_mmd_match, actions.shape[1])
        raw_actor_actions = raw_actor_actions.view(state.shape[0], self.num_samples_mmd_match, actions.shape[1])

        if self.kernel_choice == 'laplacian':
            mmd_loss = self.mmd_loss_laplacian(raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma)
        elif self.kernel_choice == 'gaussian':
            mmd_loss = self.mmd_loss_gaussian(raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma)

        action_divergence = ((sampled_actions - actor_samples) ** 2).sum(-1)
        raw_action_divergence = ((raw_sampled_actions - raw_actor_actions) ** 2).sum(-1)

        q_val1, q_val1_var = self.qf1.multiple(state, actor_samples[:, 0, :], with_var=True)
        q_val2, q_val2_var = self.qf2.multiple(state, actor_samples[:, 0, :], with_var=True)

        if self.policy_update_style == '0':
            policy_loss = torch.min(q_val1, q_val2)[:, 0]
        elif self.policy_update_style == '1':
            policy_loss = torch.mean(q_val1, q_val2)[:, 0]

        with torch.no_grad():
            q_var = q_val1_var + q_val2_var
            if self.var_Pi:
                weight = self._get_weight(q_var, self.beta).squeeze()
            else:
                weight = 1.

        if self._n_train_steps_total >= 40000:
            # Now we can update the policy
            if self.mode == 'auto':
                policy_loss = (-policy_loss * weight + self.log_alpha.exp() * (
                            mmd_loss - self.target_mmd_thresh)).mean()
            else:
                policy_loss = (-policy_loss * weight + 100 * mmd_loss).mean()
        else:
            if self.mode == 'auto':
                policy_loss = (self.log_alpha.exp() * (mmd_loss - self.target_mmd_thresh)).mean()
            else:
                policy_loss = 100 * mmd_loss.mean()
        q_value_std = q_var.mean().item()
        return policy_loss, batch_entropy, q_value_std

    def _critic_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ):
        with torch.no_grad():
            # Duplicate state 10 times (10 is a hyperparameter chosen by BCQ)
            state_rep = next_state.unsqueeze(1).repeat(1, 10, 1).view(next_state.shape[0] * 10, next_state.shape[1])

            # Compute value of perturbed actions sampled from the VAE
            action_rep = self.policy(state_rep)[0]
            target_qf1, target_qf1_var = self.target_qf1.multiple(state_rep, action_rep, with_var=True)
            target_qf2, target_qf2_var = self.target_qf2.multiple(state_rep, action_rep, with_var=True)

            # Soft Clipped Double Q-learning
            target_Q = 0.75 * torch.min(target_qf1, target_qf2) + 0.25 * torch.max(target_qf1, target_qf2)
            target_Q = target_Q.view(next_state.shape[0], -1).max(1)[0].view(-1, 1)
            target_Q = self.reward_scale * reward + (1.0 - done) * self.discount * target_Q
            target_Q_var = (target_qf1_var + target_qf2_var).view(next_state.shape[0], -1).max(1)[0].view(-1, 1)
            weight = self._get_weight(target_Q_var, self.beta)

        qf1_pred = self.qf1.sample(state, action)
        qf2_pred = self.qf2.sample(state, action)
        if self.use_exp_penalty:
            qf1_loss = ((qf1_pred - target_Q.detach()) * weight.detach()).pow(2).mean() + self.q_penalty * (
                        torch.nn.functional.relu(qf1_pred) * torch.exp(target_Q_var.data)).mean()
            qf2_loss = ((qf2_pred - target_Q.detach()) * weight.detach()).pow(2).mean() + self.q_penalty * (
                        torch.nn.functional.relu(qf2_pred) * torch.exp(target_Q_var.data)).mean()
        else:
            qf1_loss = ((qf1_pred - target_Q.detach()) * weight.detach()).pow(2).mean() + self.q_penalty * (
                        torch.nn.functional.relu(qf1_pred) * target_Q_var.data).mean()
            qf2_loss = ((qf2_pred - target_Q.detach()) * weight.detach()).pow(2).mean() + self.q_penalty * (
                        torch.nn.functional.relu(qf2_pred) * target_Q_var.data).mean()

        return qf1_loss,qf2_loss

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        state, action, reward, next_state, done = [arr.to(self.device) for arr in batch]
        # Usually updates are done in the following order: critic -> actor -> alpha
        # But we found that EDAC paper uses reverse (which gives better results)
        """
        Behavior clone a policy
        """
        recon, mean, std = self.vae(state, action)

        # 确保 recon, mean 和 std 都在同一个设备上
        recon = recon.to(self.device)
        mean = mean.to(self.device)
        std = std.to(self.device)

        # 创建 MSE 损失函数实例
        mse_loss_fn = torch.nn.MSELoss()

        # 调用损失函数，计算重构损失 recon 和 action
        recon_loss = mse_loss_fn(recon, action)

        # 计算 KL 散度损失
        kl_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()

        # 计算最终的 VAE 损失
        vae_loss = recon_loss + 0.5 * kl_loss

        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        # Actor update
        actor_loss, actor_batch_entropy, q_policy_std = self._actor_loss(state,action)
        self.policy_optimizer.zero_grad()
        if self.mode == 'auto':
            actor_loss.backward()
        self.policy_optimizer.step()

        # Critic update
        qf1_loss,qf2_loss = self._critic_loss(state, action, reward, next_state, done)
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        # alpha-lagrange update
        lagrange_loss, actor_batch_entropy, q_policy_std = self._actor_loss(state, action)
        if self.mode == 'auto':
            self.alpha_optimizer.zero_grad()
            (-lagrange_loss).backward()
            self.alpha_optimizer.step()
            self.log_alpha.data.clamp_(min=-5.0, max=10.0)
            self.alpha = self.log_alpha.exp().detach()

        #  Target networks soft update
        with torch.no_grad():
            soft_update(self.target_qf1, self.qf1, tau=self.tau)
            soft_update(self.target_qf2, self.qf2, tau=self.tau)
            # for logging, Q-ensemble std estimate with the random actions:
            # a ~ U[-max_action, max_action]
            max_action = self.policy.max_action
            random_actions = -max_action + 2 * max_action * torch.rand_like(action)

            q_random_std = self.qf1(state, random_actions).std(0).mean().item()

        self._n_train_steps_total += 1

        update_info = {
            "qf1_loss": qf1_loss.item(),
            "qf2_loss": qf2_loss.item(),
            "actor_loss": actor_loss.item(),
            "vae_loss":vae_loss.item(),
            "batch_entropy": actor_batch_entropy,
            "alpha": self.alpha.item(),
            "q_policy_std": q_policy_std,
            "q_random_std": q_random_std,
        }
        return update_info

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "critic_optim": self.critic_optimizer.state_dict(),
            "alpha_optim": self.alpha_optimizer.state_dict(),
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optim"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optim"])
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optim"])
        self.log_alpha.data[0] = state_dict["log_alpha"]
        self.alpha = self.log_alpha.exp().detach()


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: TanhGaussianPolicy, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action , agent_info = actor.get_action(state, deterministic=True)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.array(episode_rewards)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    wandb_init(asdict(config))

    # data, evaluation, env setup
    eval_env = wrap_env(gym.make(config.env_name))
    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]

    d4rl_dataset = d4rl.qlearning_dataset(eval_env)

    if config.normalize_reward:
        modify_reward(d4rl_dataset, config.env_name)

    buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=config.buffer_size,
        device=config.device,
    )
    buffer.load_d4rl_dataset(d4rl_dataset)

    # Actor & Critic setup
    qf1 = FlattenDropout_Mlp(
        input_size=state_dim + action_dim,
        output_size=1,
        hidden_sizes=[256, 256, ],
        drop_rate=config.drop_rate,
        spectral_norm=config.SN
    )
    qf2 = FlattenDropout_Mlp(
        input_size=state_dim + action_dim,
        output_size=1,
        hidden_sizes=[256, 256, ],
        drop_rate=config.drop_rate,
        spectral_norm=config.SN
    )
    target_qf1 = FlattenDropout_Mlp(
        input_size=state_dim + action_dim,
        output_size=1,
        hidden_sizes=[256, 256, ],
        drop_rate=config.drop_rate,
        spectral_norm=config.SN
    )
    target_qf2 = FlattenDropout_Mlp(
        input_size=state_dim + action_dim,
        output_size=1,
        hidden_sizes=[256, 256, ],
        drop_rate=config.drop_rate,
        spectral_norm=config.SN
    )
    policy = TanhGaussianPolicy(
        obs_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=[256, 256, ],
    )
    vae_policy = VAEPolicy(
        obs_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=[750, 750],
        latent_dim=action_dim * 2,
    )
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=config.actor_learning_rate)
    vae_policy_optimizer = torch.optim.Adam(vae_policy.parameters(), lr=config.actor_learning_rate)
    qf1_optimizer = torch.optim.Adam(qf1.parameters(), lr=config.critic_learning_rate)
    qf2_optimizer = torch.optim.Adam(qf2.parameters(), lr=config.critic_learning_rate)
    trainer = UWAC(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        vae=vae_policy,
        policy_optimizer=policy_optimizer,
        vae_policy_optimizer=vae_policy_optimizer,
        qf1_optimizer=qf1_optimizer,
        qf2_optimizer=qf2_optimizer,
        gamma=config.gamma,
        tau=config.tau,
        alpha_learning_rate=config.alpha_learning_rate,
        device=config.device,
    )
    # saving config to the checkpoint
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    total_updates = 0.0
    for epoch in trange(config.num_epochs, desc="Training"):
        # training
        for _ in trange(config.num_updates_on_epoch, desc="Epoch", leave=False):
            batch = buffer.sample(config.batch_size)
            update_info = trainer.update(batch)

            if total_updates % config.log_every == 0:
                wandb.log({"epoch": epoch, **update_info})

            total_updates += 1

        # evaluation
        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            eval_returns = eval_actor(
                env=eval_env,
                actor=policy,
                n_episodes=config.eval_episodes,
                seed=config.eval_seed,
                device=config.device,
            )
            eval_log = {
                "eval/reward_mean": np.mean(eval_returns),
                "eval/reward_std": np.std(eval_returns),
                "epoch": epoch,
            }
            if hasattr(eval_env, "get_normalized_score"):
                normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0
                eval_log["eval/normalized_score_mean"] = np.mean(normalized_score)
                eval_log["eval/normalized_score_std"] = np.std(normalized_score)

            wandb.log(eval_log)

            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"{epoch}.pt"),
                )

    wandb.finish()


if __name__ == "__main__":
    train()
