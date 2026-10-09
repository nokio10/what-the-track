import ast
import random
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "game"))

from track_order import mix_player_questions, player_blocks  # noqa: E402


def track(track_id):
    return f"https://music.yandex.ru/track/{track_id}"


class TrackOrderTests(unittest.TestCase):
    def test_blocks_are_assigned_before_deduplication(self):
        urls = [track(i) for i in (1, 2, 1, 3)]
        self.assertEqual(player_blocks(urls, 2), {"1": 0, "2": 0, "3": 1})

    def test_validation(self):
        for count in (-1, True, 2.5, "2", None):
            with self.subTest(count=count), self.assertRaises(ValueError):
                player_blocks([], count)
        cases = [
            ([track(1)], 2),
            ([], 2),
            (["https://music.yandex.ru/album/1"] * 2, 2),
        ]
        for urls, count in cases:
            with self.subTest(urls=urls), self.assertRaises(ValueError):
                player_blocks(urls, count)

    def test_disabled(self):
        self.assertEqual(player_blocks(["anything"], 0), {})

    def test_preserves_questions_and_avoids_neighbours(self):
        for sizes in [(5, 5, 5), (4, 2, 1), (1, 1), (0, 3, 3), (2, 1, 1)]:
            groups = [[{"id": f"{p}-{i}"} for i in range(n)] for p, n in enumerate(sizes)]
            expected = sorted(q["id"] for g in groups for q in g)
            for seed in range(30):
                with self.subTest(sizes=sizes, seed=seed):
                    result, repeats = mix_player_questions(groups, random.Random(seed))
                    self.assertEqual(sorted(q["id"] for q in result), expected)
                    self.assertEqual(repeats, 0)
                    owners = [q["id"].split("-")[0] for q in result]
                    self.assertTrue(all(a != b for a, b in zip(owners, owners[1:])))

    def test_unavoidable_repeats_do_not_drop_questions(self):
        groups = [[{"id": i} for i in range(5)], [{"id": 9}]]
        result, repeats = mix_player_questions(groups, random.Random(1))
        self.assertEqual(len(result), 6)
        self.assertEqual(repeats, 3)

    def test_empty(self):
        self.assertEqual(mix_player_questions([[], []]), ([], 0))


class TrackOrderIntegrationTests(unittest.TestCase):
    """Настройка проходит путь от /start до generation_task в дочернем процессе."""

    def load_function(self, name, namespace):
        path = ROOT / "src" / "game" / "generator_service.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        node.decorator_list = []
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        return namespace[name]

    def call_start(self, data):
        queue = Mock()
        namespace = {
            "job_status_lock": threading.Lock(),
            "job_status": {"is_busy": False},
            "request": SimpleNamespace(json=data),
            "jsonify": lambda d: d,
            "is_valid_game_id": lambda v: True,
            "task_queue": queue,
            "player_blocks": player_blocks,
        }
        return self.load_function("start_gen", namespace)(), queue

    def test_invalid_blocks_are_rejected_before_queueing(self):
        result, queue = self.call_start({"game_id": "TEST", "urls": [track(1)], "player_count": 2})
        self.assertEqual(result[1], 400)
        queue.put.assert_not_called()

    def test_setting_survives_queue(self):
        result, queue = self.call_start(
            {"game_id": "TEST", "urls": [track(1), track(2)], "player_count": 2}
        )
        self.assertEqual(queue.put.call_args.args[0]["player_count"], 2)

    def test_setting_reaches_child_process(self):
        task = {"game_id": "TEST", "urls": [], "player_count": 3}
        target = Mock()
        namespace = {
            "_ForwardedStatus": lambda events: {},
            "generation_task": target,
            "_GIGAAM_STATE": {},
            "job_status": {},
            "log": Mock(),
        }
        self.load_function("_generation_child", namespace)(task, Mock(), 0)
        target.assert_called_once_with("TEST", None, [], 3)


if __name__ == "__main__":
    unittest.main()
