"""Protocol tests with synthetic evidence; no official comparisons are run."""
from pathlib import Path
import hashlib
import json

import pytest

import compare_runs as comparison
from dqn import DQNAgent, defaults
from train import evaluate_model, write_csv, write_json


def outcomes(rewards, seeds=None):
    seeds = range(200007, 200007 + len(rewards)) if seeds is None else seeds
    return [{'episode': i+1, 'seed': seed, 'reward': float(reward), 'length': 10,
             'terminated': True, 'truncated': False, 'max_q': 1.0}
            for i, (seed, reward) in enumerate(zip(seeds, rewards, strict=True))]


@pytest.fixture
def evidence(tmp_path):
    baseline, candidate, output = (tmp_path/name for name in ('baseline', 'candidate', 'comparison'))
    for path in (baseline, candidate, output):
        path.mkdir()
    agent = DQNAgent((4,), 2, defaults('cartpole'))
    agent.save(baseline/'policy.pt', {'episode': 1})
    agent.save(candidate/'policy.pt', {'episode': 2, 'development_mean': 1600., 'development_seed': 40000})
    baseline_hash = hashlib.sha256((baseline/'policy.pt').read_bytes()).hexdigest()
    write_json(output/'experiment_plan.json', {
        'old_model_development_mean': 1542.,
        'selection_rule': 'Strictly greater development mean, frozen before final test.',
        'mspacman': {'episodes': 3, 'development_seed': 40000, 'development_episodes': 20,
                    'final_evaluation_seed': 200007, 'final_evaluation_episodes': 3}})
    write_json(output/'baseline_development20.json', {
        'checkpoint_sha256': baseline_hash, 'seed_start': 40000,
        'development': {'n': 20, 'mean': 1542.}})
    write_csv(output/'baseline_development20.csv', outcomes([1542.]*20, range(40000, 40020)))
    write_csv(candidate/'development.csv', [
        {'training_episode': 2, 'n': 20, 'mean': 1600.},
        {'training_episode': 3, 'n': 20, 'mean': 1500.}])
    write_json(candidate/'config.json', {'development_seed': 40000, 'development_episodes': 20,
               'evaluation_seed': 200007, 'final_episodes': 3, 'requested_episodes': 3})
    return baseline, candidate, output


def freeze(evidence):
    baseline, candidate, output = evidence
    return comparison.freeze_selection(baseline, candidate, output, output/'experiment_plan.json')


def test_decision_freezes_from_development_before_any_final_results(evidence):
    selected = freeze(evidence)
    assert selected['selection']['candidate_selected']
    assert selected['selection']['baseline_development_mean'] == 1542
    assert selected['selection']['candidate_development_mean'] == 1600
    assert (evidence[2]/'selection_decision.json').exists()
    write_csv(evidence[1]/'evaluation.csv', outcomes([9999, 9999, 9999]))
    with pytest.raises(FileExistsError, match='evaluation'):
        freeze(evidence)


def test_comparison_uses_matched_seed_order_and_never_selects_using_test_scores(evidence, monkeypatch):
    baseline, candidate, output = evidence
    agent = DQNAgent((4,), 2, defaults('cartpole'))
    agent.save(candidate/'policy.pt', {'episode': 3, 'development_mean': 1542., 'development_seed': 40000})
    write_csv(candidate/'development.csv', [
        {'training_episode': 2, 'n': 20, 'mean': 1500.},
        {'training_episode': 3, 'n': 20, 'mean': 1542.}])
    frozen = freeze(evidence)
    assert not frozen['selection']['candidate_selected']  # ties retain baseline
    write_csv(candidate/'evaluation.csv', outcomes([10, 30, 20], [200009, 200007, 200008]))
    calls = []
    def evaluate(agent, episodes, seed, epsilon=0, **kwargs):
        calls.append((episodes, seed, epsilon))
        return outcomes([2, 2, 2] if epsilon == 1 else [5, 6, 7]), []
    monkeypatch.setattr(comparison, 'evaluate_model', evaluate)
    result = comparison.compare_runs(baseline, candidate, output, episodes=3)
    assert result['comparisons']['candidate_minus_baseline']['mean_difference'] == 14
    assert result['comparisons']['candidate_minus_random']['mean_difference'] == 18
    assert not result['selection']['candidate_selected']
    assert calls == [(3, 200007, 0.0), (3, 200007, 1.0)]
    assert result == json.loads((output/'comparison.json').read_text())
    assert result['baseline']['mean_reward'] == 6
    assert result['candidate']['std_reward'] == 10
    assert result['weights_unchanged']


def test_mismatched_final_seeds_fail_before_new_evaluation(evidence, monkeypatch):
    freeze(evidence)
    write_csv(evidence[1]/'evaluation.csv', outcomes([1, 2, 3], [200007, 200008, 99999]))
    monkeypatch.setattr(comparison, 'evaluate_model', lambda *a, **k: pytest.fail('must validate first'))
    with pytest.raises(ValueError, match='seed'):
        comparison.compare_runs(*evidence, episodes=3)


def test_changed_checkpoint_invalidates_frozen_decision(evidence):
    freeze(evidence)
    with (evidence[1]/'policy.pt').open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='changed'):
        comparison.compare_runs(*evidence, episodes=3)


def test_freeze_rejects_wrong_or_incomplete_development_protocol(evidence):
    write_csv(evidence[1]/'development.csv', [{'training_episode': 2, 'n': 20, 'mean': 1600.}])
    with pytest.raises(ValueError, match='complete'):
        freeze(evidence)


def test_real_frozen_evaluation_preserves_files_and_reproduces_candidate(evidence):
    baseline, candidate, output = evidence
    freeze(evidence)
    agent, _ = DQNAgent.load(candidate/'policy.pt')
    rows, _ = evaluate_model(agent, 3, 200007)
    write_csv(candidate/'evaluation.csv', rows)
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (baseline/'policy.pt', candidate/'policy.pt')}
    result = comparison.compare_runs(baseline, candidate, output, episodes=3)
    assert result['comparisons']['candidate_minus_baseline']['mean_difference'] == 0
    assert result['evaluation'] == {'episodes': 3, 'seed_start': 200007, 'seed_end': 200009}
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    assert (output/'baseline_evaluation.csv').exists() and (output/'random_evaluation.csv').exists()


def test_cartpole_summary_keeps_primary_and_checks_matched_fresh_seeds(tmp_path):
    root, primary = tmp_path/'controls', tmp_path/'cartpole'
    primary.mkdir()
    write_json(primary/'config.json', {'seed': 7, 'task': 'cartpole'})
    write_json(primary/'metrics.json', {'training_episodes': 1500,
                                      'selected_checkpoint': {'episode': 1300}})
    rows = [{**row, 'length': 500, 'terminated': False, 'truncated': True}
            for row in outcomes([500.]*500, range(200007, 200507))]
    write_csv(root/'seed7_evaluation'/'evaluation.csv', rows)
    for seed in (8, 9):
        write_json(root/f'seed{seed}'/'config.json', {'seed': seed, 'task': 'cartpole',
                   'evaluation_seed': 200007, 'final_episodes': 500})
        write_json(root/f'seed{seed}'/'metrics.json', {'training_episodes': 200,
                   'selected_checkpoint': {'episode': 150}})
        write_csv(root/f'seed{seed}'/'evaluation.csv', rows)
    result = comparison.summarize_cartpole_controls(root)
    assert result['primary_seed'] == 7
    assert [run['seed'] for run in result['runs']] == [7, 8, 9]
    assert all(run['successes'] == 500 and run['mean_reward'] == 500 for run in result['runs'])
    assert result['runs'][0]['selected_episode'] == 1300
    assert result == json.loads((root/'summary.json').read_text())
    write_csv(root/'seed9'/'evaluation.csv', [{**rows[0], 'seed': 123}, *rows[1:]])
    with pytest.raises(ValueError, match='seed'):
        comparison.summarize_cartpole_controls(root)


def test_cli_freeze_and_protocol_mismatch(evidence, capsys):
    baseline, candidate, output = evidence
    comparison.main(['--baseline-run', str(baseline), '--new-run', str(candidate),
                     '--output', str(output), '--freeze-only'])
    assert json.loads(capsys.readouterr().out)['selection']['candidate_selected']
    with pytest.raises(ValueError, match='protocol'):
        comparison.compare_runs(*evidence, episodes=500)
    with pytest.raises(FileExistsError, match='already'):
        freeze(evidence)
    with pytest.raises(SystemExit):
        comparison.main([])


def test_freeze_rejects_wrong_selected_checkpoint_and_changed_development(evidence):
    agent = DQNAgent((4,), 2, defaults('cartpole'))
    agent.save(evidence[1]/'policy.pt', {'episode': 3, 'development_mean': 1500., 'development_seed': 40000})
    with pytest.raises(ValueError, match='best development'):
        freeze(evidence)
    agent.save(evidence[1]/'policy.pt', {'episode': 2, 'development_mean': 1600., 'development_seed': 40000})
    freeze(evidence)
    with (evidence[1]/'development.csv').open('a') as handle:
        handle.write('\n')
    with pytest.raises(ValueError, match='changed'):
        comparison.compare_runs(*evidence, episodes=3)


@pytest.mark.parametrize('change', [{'terminated': 'invalid'}, {'reward': float('nan')},
                                   {'terminated': False, 'truncated': False}])
def test_rejects_invalid_evaluation_records(change):
    with pytest.raises(ValueError):
        comparison._evaluation([{**outcomes([1])[0], **change}], 1, 200007)
