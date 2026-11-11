"""Behavioral checks for controlled RL experiments, without game training."""
import hashlib
import json

import numpy as np
import pytest
import torch

import scientific_validation as scientific


def test_known_solution_and_discounted_return():
    np.testing.assert_allclose(scientific.known_solution(0.95, 3), [[2.85, 0], [3, 0]])
    np.testing.assert_allclose(scientific.known_solution(0, 1), [[0, 0], [1, 0]])
    assert scientific.discounted_return([1, 2, 3], 0.5) == pytest.approx(2.75)
    assert scientific.discounted_return([1, 2, 3], 0) == 1
    with pytest.raises(ValueError):
        scientific.known_solution(1.2, 1)
    with pytest.raises(ValueError):
        scientific.discounted_return([float('nan')], 0.9)


def test_image_interventions_remove_history_or_information_without_mutation():
    state = np.arange(48, dtype=np.uint8).reshape(4, 3, 4)
    before = state.copy()
    history = scientific.repeat_latest_frame(state)
    constant = scientific.constant_midgray(state)
    np.testing.assert_array_equal(state, before)
    np.testing.assert_array_equal(history, np.repeat(state[-1:], 4, axis=0))
    assert history.dtype == state.dtype and not np.shares_memory(state, history)
    assert np.all(constant == 128) and constant.dtype == np.uint8
    with pytest.raises(ValueError):
        scientific.repeat_latest_frame(np.zeros(4))


def calibration_row(seed, prediction, realized, terminated=True):
    return {'seed': seed, 'predicted_q': prediction, 'discounted_return': realized,
            'terminated': terminated, 'truncated': not terminated,
            'tail_upper_bound': 0.0 if terminated else 0.2}


def test_calibration_reports_signed_bias_error_and_censoring_honestly():
    result = scientific.summarize_calibration([
        calibration_row(1, 3, 2), calibration_row(2, 2, 4, False)])
    assert result['mean_signed_error'] == pytest.approx(-0.5)
    assert result['mae'] == pytest.approx(1.5)
    assert result['rmse'] == pytest.approx(np.sqrt(2.5))
    assert result['pearson_correlation'] == pytest.approx(-1)
    assert result['censored_episodes'] == 1
    assert result['maximum_tail_upper_bound'] == pytest.approx(0.2)
    constant = scientific.summarize_calibration([calibration_row(1, 3, 2), calibration_row(2, 4, 2)])
    assert constant['pearson_correlation'] is None
    with pytest.raises(ValueError):
        scientific.summarize_calibration([])


def test_real_dqn_update_recovers_known_values_but_zero_rate_control_does_not():
    torch.set_num_threads(1)
    learned = scientific.analytic_case(0.95, 3, steps=500)
    control = scientific.analytic_case(0.95, 3, steps=100, learning_rate=0)
    assert learned['max_absolute_error'] < 0.02
    assert learned['passed']
    assert not control['passed'] and control['max_absolute_error'] > 0.1
    assert control['weights_unchanged']
    assert not learned['weights_unchanged']


class EndingEnvironment:
    def __init__(self, truncated):
        self.truncated = truncated
        self.closed = False

    def reset(self, seed):
        return np.zeros(4, np.float32), {}

    def step(self, action):
        return np.ones(4, np.float32), 1.0, not self.truncated, self.truncated, {}

    def close(self):
        self.closed = True


@pytest.mark.parametrize('truncated', [False, True])
def test_observed_ending_uses_actual_target_code_and_final_successor(monkeypatch, truncated):
    from dqn import DQNAgent, defaults
    agent = DQNAgent((4,), 2, defaults('cartpole'))
    env = EndingEnvironment(truncated)
    monkeypatch.setattr(scientific, 'make_env', lambda *args, **kwargs: env)
    monkeypatch.setattr(scientific, 'q_values', lambda a, s, target=False: np.array([10., 7.]) if target else np.array([2., 1.]))
    rows, endings, states = scientific.calibration_rollouts(agent, episodes=1, seed=99)
    assert env.closed
    assert rows[0]['discounted_return'] == 1
    assert rows[0]['predicted_q'] == 2
    assert rows[0]['truncated'] is truncated
    assert rows[0]['tail_upper_bound'] == pytest.approx(19 if truncated else 0)
    assert endings[0]['target'] == pytest.approx(10.5 if truncated else 1)
    assert endings[0]['matches_expected']
    assert scientific.summarize_endings(endings)['truncation']['count'] == int(truncated)


def test_scientific_workflow_preserves_real_frozen_cartpole_checkpoint(tmp_path):
    from dqn import DQNAgent, defaults
    agent = DQNAgent((4,), 2, defaults('cartpole'))
    checkpoint = tmp_path/'policy.pt'
    agent.save(checkpoint, {'purpose': 'untrained test fixture'})
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = scientific.run_scientific_validation(tmp_path, episodes=2, analytic_steps=500)
    assert result == json.loads((tmp_path/'scientific_validation.json').read_text())
    assert result['checkpoint_sha256'] == digest
    assert result['weights_unchanged']
    assert result['calibration']['episodes'] == 2
    assert result['sampled_calibration']['observations'] > 2
    assert result['analytic_learning']['all_known_solutions_passed']
    assert result['ending_target_audit']['terminal']['count'] == 2
    assert result['pixel_interventions'] is None
    assert (tmp_path/'calibration.csv').exists() and (tmp_path/'ending_targets.csv').exists()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize('episodes,steps,seed', [(0, 500, 1), (1, 0, 1), (1, 500, -1)])
def test_invalid_requests_fail_before_loading(tmp_path, episodes, steps, seed):
    with pytest.raises(ValueError):
        scientific.run_scientific_validation(tmp_path, episodes=episodes, analytic_steps=steps, seed=seed)


def test_sampled_calibration_censors_near_time_limit_but_keeps_terminal_states():
    rewards = np.ones(150)
    predictions = np.full(150, 20.)
    timed = scientific.sample_calibration_states(1, predictions, rewards, 0.95, False, True)
    assert len(timed) == 50
    assert all(row['remaining_horizon'] >= 100 for row in timed)
    assert max(row['tail_upper_bound'] for row in timed) <= 0.95**100 / 0.05 + 1e-12
    assert timed[0]['step'] == 0
    terminal = scientific.sample_calibration_states(1, predictions, rewards, 0.95, True, False)
    assert terminal[-1]['remaining_horizon'] == 1
    assert all(row['tail_upper_bound'] == 0 for row in terminal)
    assert terminal[-1]['discounted_return'] == 1


def test_real_pixel_workflow_keeps_matching_seeds_and_reports_null_result(tmp_path, monkeypatch):
    import train
    from dqn import DQNAgent, defaults, make_env
    torch.set_num_threads(1)
    agent = DQNAgent((4, 88, 80), 9, defaults('mspacman'))
    agent.save(tmp_path/'policy.pt', {'purpose': 'untrained pixel test fixture'})

    def tiny_environment(task, frame_stack=4):
        return make_env(task, frame_stack=frame_stack, max_episode_steps=2)

    monkeypatch.setattr(scientific, 'make_env', tiny_environment)
    monkeypatch.setattr(train, 'make_env', tiny_environment)
    result = scientific.run_scientific_validation(tmp_path, episodes=2, analytic_steps=500)
    assert result['weights_unchanged']
    assert result['sampled_calibration'] is None  # fewer than 100 steps before truncation
    assert result['ending_target_audit']['truncation']['count'] == 2
    conditions = result['pixel_interventions']['conditions']
    for name in ('repeat_latest', 'constant128'):
        delta = conditions[name]['normal_minus_condition']
        assert delta['n_pairs'] == 2
        assert delta['mean_difference'] == 0
        assert not delta['positive_difference_supported']
        assert (tmp_path/f'scientific_{name}.csv').exists()
