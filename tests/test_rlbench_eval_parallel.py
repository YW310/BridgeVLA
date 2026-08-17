import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'finetune'
    / 'RLBench'
    / 'eval_parallel.py'
)
SPEC = importlib.util.spec_from_file_location('rlbench_eval_parallel', MODULE_PATH)
eval_parallel = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = eval_parallel
SPEC.loader.exec_module(eval_parallel)


class RLBenchEvalParallelTest(unittest.TestCase):
    def test_default_suite_contains_all_eighteen_tasks(self):
        self.assertEqual(len(eval_parallel.RLBENCH_TASKS), 18)

    def test_gpu_ids_are_unique(self):
        self.assertEqual(eval_parallel.parse_gpu_ids('0,2,5'), ('0', '2', '5'))
        with self.assertRaisesRegex(Exception, 'must be unique'):
            eval_parallel.parse_gpu_ids('0,0')

    def test_reads_exactly_one_expected_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'eval_results.csv'
            with path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        'task',
                        'success rate',
                        'length',
                        'total_transitions',
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        'task': 'open_drawer',
                        'success rate': '88.0',
                        'length': '5.0',
                        'total_transitions': '125',
                    }
                )
            row = eval_parallel.read_task_result(path, 'open_drawer')
            self.assertEqual(row['success rate'], 88.0)
            with self.assertRaisesRegex(RuntimeError, 'expected close_jar'):
                eval_parallel.read_task_result(path, 'close_jar')


if __name__ == '__main__':
    unittest.main()
