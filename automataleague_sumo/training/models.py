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


class BoundedLocNormalScale(AddStateIndependentNormalScale):
    """Add the state-independent scale, and keep the mean in a finite range.

    Subclassed rather than inserted as its own ``Sequential`` entry so the module
    indices and the ``state_independent_scale`` parameter name are unchanged: a
    new entry would shift every key and no existing checkpoint would load.

    The policy is a TanhNormal, so its log-prob carries a ``-log(1 - a^2)``
    jacobian that grows without limit as the action approaches the bound. Nothing
    bounded ``loc``, and three runs of this project died on that: the importance
    ratio is ``exp(new_log_prob - old_log_prob)``, which overflows float32 above
    88, and a diagnostic dump caught a stored log-prob of 3427 while every other
    quantity in the network was healthy (entropy 8.39, sigma 0.43, explained
    variance 0.80). A log-prob that size needs a mean tens of units outside the
    tanh range, where it commands no additional motion whatsoever.

    ``limit * tanh(x / limit)`` and NOT ``x.clamp(-limit, limit)``. clamp has zero
    gradient outside its range, so it would stop pulling the mean back at exactly
    the point that matters. That is not hypothetical: clamping the importance
    ratio for the same reason silently destroyed a 40M-frame run by deleting the
    policy gradient on every saturated sample. This is smooth, its gradient is
    positive everywhere, and it is near-inert in the range a healthy policy uses
    (measured on a 290M-frame checkpoint: median |loc| 0.36, 99th percentile 2.07,
    max 3.62 against the default limit of 5).

    ``limit=None`` disables the bound entirely, which is what a checkpoint saved
    before ``network.max_loc`` existed gets. Those checkpoints are replayed for
    renders and the round robin, and a policy that behaves differently on replay
    than it did when trained would invalidate every comparison already made.
    """

    def __init__(self, *args, limit: float | None, **kwargs):
        super().__init__(*args, **kwargs)
        if limit is not None and limit <= 0:
            raise ValueError(f"network.max_loc must be > 0 or None, got {limit}")
        self.limit = None if limit is None else float(limit)

    def forward(self, loc, *others):
        if self.limit is not None:
            loc = self.limit * torch.tanh(loc / self.limit)
        return super().forward(loc, *others)


def build_actor(cfg, robot, device):
    """Rebuild the actor from a robot spec alone, with no live GPU env.

    Rendering and head-to-head evaluation run on CPU, where constructing a Warp env
    just to read two integers off its specs is not possible. Both widths are derived
    from the same functions the real env uses, so a stub that disagreed with the
    trained checkpoint would fail loudly at load_state_dict rather than quietly
    producing a differently-shaped policy.
    """
    from torchrl.data import Bounded, Composite, Unbounded

    from automataleague_sumo.envs.sumo.observation import observation_dim

    obs_dim, act_dim = observation_dim(robot), robot.action_dim

    class _Stub:
        batch_size = torch.Size([1])
        observation_spec = Composite(
            observation=Unbounded(shape=(1, obs_dim), device=device), shape=(1,))
        action_spec = Bounded(
            low=-torch.ones(1, act_dim, device=device),
            high=torch.ones(1, act_dim, device=device), device=device)

    actor, _ = make_ppo_models(cfg, _Stub(), device)
    return actor


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
        # `.get` defaulting to None, not attribute access: checkpoints saved
        # before max_loc existed carry their own config and are reloaded for
        # rendering and the round robin. They must both keep loading AND keep
        # behaving exactly as they did when trained, so they get no bound.
        BoundedLocNormalScale(
            num_outputs, scale_lb=1e-8,
            limit=cfg.network.get("max_loc", None)).to(device),
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
