"""Behavioral checks for the shared DQN implementation (no long training)."""
import importlib
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import torch


@pytest.fixture
def core():
    return importlib.import_module("dqn")


def test_defaults_and_immutable_configuration(core):
    cart, pac = core.defaults("cartpole"), core.defaults("mspacman")
    assert (cart.gamma, pac.gamma) == (0.95, 0.99)
    assert cart.reward_clip is False and pac.reward_clip is True
    with pytest.raises(FrozenInstanceError):
        cart.gamma = 0.2


@pytest.mark.parametrize("change", [dict(task="unknown"), dict(batch_size=0),
    dict(gamma=1.1), dict(epsilon_end=-0.1), dict(learning_rate=float("nan")),
    dict(buffer_size=2), dict(frame_stack=0), dict(train_frequency=0)])
def test_configuration_rejects_invalid_values(core, change):
    with pytest.raises(ValueError):
        replace(core.defaults("cartpole"), **change)


def test_preprocessing_matches_assignment_without_modifying_input(core):
    rgb = np.full((210, 160, 3), 255, dtype=np.uint8)
    rgb[1, 0] = [210, 164, 74]
    before = rgb.copy()
    result = core.preprocess_observation(rgb)
    assert result.shape == (88, 80) and result.dtype == np.uint8
    assert result[0, 0] == 0 and result[1, 1] == 255
    assert np.array_equal(rgb, before)
    with pytest.raises(ValueError):
        core.preprocess_observation(np.zeros((3, 3)))


@pytest.mark.parametrize("task,shape,actions", [
    ("cartpole", (4,), 2), ("mspacman", (4, 88, 80), 9)])
def test_real_environments_and_networks(core, task, shape, actions):
    env = core.make_env(task, seed=19)
    try:
        state, _ = env.reset(seed=19)
        assert state.shape == shape and env.action_space.n == actions
        if task == "cartpole":
            assert env.spec.id == "CartPole-v1"
            assert env.spec.max_episode_steps == 500
        else:
            assert env.spec.id == "ALE/MsPacman-v5"
            assert state.dtype == np.uint8
        network = core.QNetwork(shape, actions)
        output = network(torch.as_tensor(state).unsqueeze(0))
        assert output.shape == (1, actions) and torch.isfinite(output).all()
        assert len(env.step(0)) == 5
    finally:
        env.close()


def test_environment_counterfactuals_are_explicit(core):
    env = core.make_env("cartpole", max_episode_steps=1)
    try:
        env.reset(seed=9)
        _, _, terminated, truncated, _ = env.step(0)
        assert truncated and not terminated
    finally:
        env.close()
    with pytest.raises(ValueError):
        core.make_env("other")
    with pytest.raises(ValueError):
        core.make_env("mspacman", repeat_action_probability=-0.1)


def test_replay_keeps_recent_transitions_and_copies_inputs(core):
    replay = core.ReplayBuffer(3, seed=1)
    for value in range(5):
        state = np.full(4, value, dtype=np.float32)
        replay.add(state, 0, value, state + 1, value == 4, value == 3)
        state[:] = -100
    batch = replay.sample(3, "cpu")
    assert len(replay) == 3
    assert set(batch.states[:, 0].tolist()) == {2, 3, 4}
    assert torch.equal(batch.next_states, batch.states + 1)
    assert batch.actions.dtype == torch.long
    assert batch.terminated.sum() == 1 and batch.truncated.sum() == 1


def test_replay_validation_and_reproducible_sampling(core):
    with pytest.raises(ValueError):
        core.ReplayBuffer(0)
    a, b = core.ReplayBuffer(3, 7), core.ReplayBuffer(3, 7)
    with pytest.raises(ValueError):
        a.sample(1)
    for value in range(3):
        for buffer in (a, b):
            buffer.add(np.array([value]), 0, 1, np.array([value + 1]), False)
    assert torch.equal(a.sample(2).states, b.sample(2).states)
    with pytest.raises(ValueError):
        a.add(np.zeros(1), -1, 0, np.zeros(1), False)
    with pytest.raises(ValueError):
        a.add(np.zeros(1), 0, float("nan"), np.zeros(1), False)
    with pytest.raises(ValueError):
        a.add(np.zeros(2), 0, 0, np.zeros(2), False)
    with pytest.raises(ValueError):
        a.add(np.array([float("nan")]), 0, 0, np.zeros(1), False)


def test_terminal_target_and_time_limit_bootstrapping(core):
    rewards = torch.tensor([1.0, 1.0, 1.0])
    terminated = torch.tensor([True, False, False])
    next_values = torch.tensor([100.0, 10.0, 10.0], requires_grad=True)
    target = core.bellman_targets(rewards, terminated, next_values, 0.95)
    assert target.tolist() == pytest.approx([1, 10.5, 10.5])
    assert not target.requires_grad
    assert core.bellman_targets(rewards, terminated, next_values, 0).tolist() == [1]*3


def make_agent_and_batch(core):
    cfg = replace(core.defaults("cartpole"), batch_size=4)
    agent = core.DQNAgent((4,), 2, cfg)
    buffer = core.ReplayBuffer(4)
    for action in [0, 1, 0, 1]:
        buffer.add(np.zeros(4, np.float32), action, 2, np.ones(4, np.float32), True)
    return agent, buffer.sample(4)


def test_learning_updates_online_but_freezes_target_until_sync(core):
    agent, batch = make_agent_and_batch(core)
    original_target = [p.clone() for p in agent.target.parameters()]
    original_online = [p.clone() for p in agent.online.parameters()]
    first = agent.update(batch)
    for _ in range(20):
        last = agent.update(batch)
    assert last["loss"] < first["loss"]
    assert set(first) == {"loss", "td_abs", "grad_norm"}
    assert any(not torch.equal(p, old) for p, old in zip(agent.online.parameters(), original_online))
    assert all(torch.equal(p, old) for p, old in zip(agent.target.parameters(), original_target))
    assert all(p.grad is None and not p.requires_grad for p in agent.target.parameters())
    agent.sync_target()
    assert all(torch.equal(p, q) for p, q in zip(agent.online.parameters(), agent.target.parameters()))


def test_epsilon_greedy_and_checkpoint_round_trip(core, tmp_path):
    agent, batch = make_agent_and_batch(core)
    agent.update(batch)
    state = np.zeros(4, np.float32)
    rng = np.random.default_rng(3)
    expected = agent.online(torch.tensor(state)[None]).argmax().item()
    assert agent.act(state, 0, rng) == expected
    assert set(agent.act(state, 1, rng) for _ in range(50)) == {0, 1}
    with pytest.raises(ValueError):
        agent.act(state, 2, rng)
    path = tmp_path / "policy.pt"
    agent.save(path, {"episode": 12})
    loaded, metadata = core.DQNAgent.load(path)
    assert metadata == {"episode": 12}
    assert loaded.config == agent.config
    assert torch.equal(agent.online(torch.tensor(state)[None]), loaded.online(torch.tensor(state)[None]))
    assert loaded.optimizer.state_dict()["state"]


def test_network_shape_validation(core):
    with pytest.raises(ValueError):
        core.QNetwork((3, 2), 2)
    with pytest.raises(ValueError):
        core.QNetwork((4,), 0)


def test_complete_short_environment_replay_update_checkpoint_flow(core, tmp_path):
    config = replace(core.defaults("cartpole"), batch_size=4)
    env = core.make_env("cartpole", seed=11)
    agent = core.DQNAgent(env.observation_space.shape, env.action_space.n, config)
    replay = core.ReplayBuffer(20)
    rng = np.random.default_rng(11)
    state, _ = env.reset(seed=11)
    try:
        for _ in range(12):
            action = agent.act(state, 0.5, rng)
            nxt, reward, term, trunc, _ = env.step(action)
            replay.add(state, action, reward, nxt, term, trunc)
            if len(replay) >= 4:
                assert np.isfinite(agent.update(replay.sample(4))["loss"])
            state = env.reset()[0] if term or trunc else nxt
        agent.sync_target()
        agent.save(tmp_path / "integration.pt")
        restored, _ = core.DQNAgent.load(tmp_path / "integration.pt")
        assert restored.act(state, 0, rng) in [0, 1]
    finally:
        env.close()
