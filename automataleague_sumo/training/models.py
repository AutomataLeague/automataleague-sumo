"""PPO actor/critic builders. Task agnostic — every dimension comes from the env."""

from __future__ import annotations

import torch
import torch.nn
from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torchrl.envs import ExplorationType
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator

_ACTIVATIONS = {
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "leaky_relu": torch.nn.LeakyReLU,
    "elu": torch.nn.ELU,
}


def get_activation(name: str):
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Valid: {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]


def make_ppo_models(cfg, train_env, device):
    """Actor (MLP -> TanhNormal ProbabilisticActor) and critic (MLP -> ValueOperator).

    One shared actor drives every row of the batch. Under self-play that batch holds
    both contestants of every duel, so this single network is simultaneously both
    wrestlers — which is only sound because the observation is written in each
    robot's own base frame and carries no absolute side identity.
    """
    obs_dim = train_env.observation_spec["observation"].shape[-1]
    action_spec = train_env.action_spec
    if train_env.batch_size:
        action_spec = action_spec[(0,) * len(train_env.batch_size)]
    num_outputs = action_spec.shape[-1]

    activation_class = get_activation(cfg.network.activation)
    hidden_sizes = list(cfg.network.hidden_sizes)

    policy_mlp = MLP(
        in_features=obs_dim, out_features=num_outputs,
        num_cells=hidden_sizes, activation_class=activation_class, device=device,
    )
    for layer in policy_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 1.0)
            layer.bias.data.zero_()

    policy_mlp = torch.nn.Sequential(
        policy_mlp,
        AddStateIndependentNormalScale(num_outputs, scale_lb=1e-8).to(device),
    )

    policy_module = ProbabilisticActor(
        TensorDictModule(module=policy_mlp, in_keys=["observation"],
                         out_keys=["loc", "scale"]),
        in_keys=["loc", "scale"],
        spec=action_spec,
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": action_spec.space.low,
            "high": action_spec.space.high,
            "tanh_loc": False,
        },
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    value_mlp = MLP(
        in_features=obs_dim, out_features=1,
        num_cells=hidden_sizes, activation_class=activation_class, device=device,
    )
    for layer in value_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.01)
            layer.bias.data.zero_()

    return policy_module, ValueOperator(value_mlp, in_keys=["observation"])
