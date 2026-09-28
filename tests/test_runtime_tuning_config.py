import os
import unittest
from unittest.mock import patch

from tests.generator_service_logic_loader import load_generator_service_logic


logic = load_generator_service_logic()
get_runtime_tuning = logic["get_runtime_tuning"]


class RuntimeTuningConfigTests(unittest.TestCase):
    def test_cpu_defaults_use_smaller_batches(self):
        with patch.dict(os.environ, {}, clear=True):
            tuning = get_runtime_tuning("cpu")

        self.assertEqual(tuning["mdx_batch_size"], 1)
        self.assertEqual(tuning["whisper_threads"], 8)
        self.assertEqual(tuning["whisper_batch_size"], 8)

    def test_gpu_defaults_keep_existing_heavier_profile(self):
        with patch.dict(os.environ, {}, clear=True):
            tuning = get_runtime_tuning("cuda")

        self.assertEqual(tuning["mdx_batch_size"], 8)
        self.assertEqual(tuning["whisper_threads"], 8)
        self.assertEqual(tuning["whisper_batch_size"], 16)

    def test_env_overrides_take_priority(self):
        with patch.dict(
            os.environ,
            {
                "MDX_BATCH_SIZE": "3",
                "WHISPER_THREADS": "12",
                "WHISPER_BATCH_SIZE": "5",
            },
            clear=True,
        ):
            tuning = get_runtime_tuning("cpu")

        self.assertEqual(tuning["mdx_batch_size"], 3)
        self.assertEqual(tuning["whisper_threads"], 12)
        self.assertEqual(tuning["whisper_batch_size"], 5)

    def test_invalid_env_values_fall_back_to_defaults(self):
        with patch.dict(
            os.environ,
            {
                "MDX_BATCH_SIZE": "oops",
                "WHISPER_THREADS": "0",
                "WHISPER_BATCH_SIZE": "-7",
            },
            clear=True,
        ):
            tuning = get_runtime_tuning("cpu")

        self.assertEqual(tuning["mdx_batch_size"], 1)
        self.assertEqual(tuning["whisper_threads"], 8)
        self.assertEqual(tuning["whisper_batch_size"], 8)


if __name__ == "__main__":
    unittest.main()
