import ast
import csv
import io
import sys
from multiprocessing import Lock
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'finetune' / 'RLBench'))
from utils.eval_reporting import MANIFEST_FIELDS, manifest_result, numeric_task_scores


def accumulator():
    # Execute the actual dependency-free accumulator classes, omitting imports
    # of YARR agent/transition types that otherwise require torch in unit tests.
    path = ROOT / 'finetune/bridgevla/libs/YARR/yarr/utils/stat_accumulator.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    classes = ast.Module(body=[node for node in tree.body if isinstance(node, ast.ClassDef)],
                         type_ignores=[])
    scope = dict(np=np, Lock=Lock, List=List, Summary=object, ReplayTransition=object,
                 ScalarSummary=lambda name, value: SimpleNamespace(name=name, value=value))
    exec(compile(classes, str(path), 'exec'), scope)
    return scope['SimpleAccumulator']()


@pytest.mark.parametrize('rewards, expected', [([100.], 100.), ([0.], 0.), ([100., 0.], 50.)])
def test_completed_episode_metrics_are_numeric_and_drained(rewards, expected):
    stats = accumulator()
    assert stats.pop() == []
    for reward in rewards:
        stats.step(SimpleNamespace(reward=reward, terminal=True, summaries=[]), True)
    values = {s.name: s.value for s in stats.pop()}
    assert values['eval_envs/return'] == expected
    assert values['eval_envs/length'] == 1
    assert stats.pop() == []


def test_no_completed_episode_is_not_reported_as_success():
    stats = accumulator()
    stats.step(SimpleNamespace(reward=0., terminal=False, summaries=[]), True)
    assert stats.pop() == []


def test_manifest_coverage_is_separate_from_policy_success():
    result = manifest_result('close_jar', 1, 1, 1)
    assert result['generated coverage'] == 100.
    assert 'success rate' not in result
    assert manifest_result('close_jar', 1, 2, 1)['generated coverage'] == 50.
    assert manifest_result('close_jar', 0, 1, 0)['generated coverage'] == 0.
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
    writer.writeheader()
    writer.writerow(result)
    assert 'success rate' not in stream.getvalue()
    assert numeric_task_scores({'close_jar': result['generated coverage']}) == {'close_jar': 100.}


def test_missing_metrics_are_not_fabricated_or_written_as_strings():
    assert numeric_task_scores({'a': 'unknown', 'b': None, 'c': float('nan'),
                                'd': float('inf'), 'e': 0., 'f': np.float32(100.)}) == {
                                    'e': 0., 'f': 100.}
    with pytest.raises(ValueError):
        manifest_result('close_jar', 0, 0, 0)


def test_eval_wires_separate_manifest_csv_and_tensorboard_namespace():
    source = (ROOT / 'finetune/RLBench/eval.py').read_text(encoding='utf-8')
    ast.parse(source)
    assert 'manifest_results.csv' in source
    assert "'manifest_coverage'" in source
    assert 'numeric_task_scores(task_scores)' in source
    shell = (ROOT / 'finetune/RLBench/eval.sh').read_text(encoding='utf-8')
    assert 'merged_manifest_results.csv' in shell
    assert '${result_filename}' in shell
