import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_SERVICE = ROOT / "generator_service.py"
DOCKER_COMPOSE = ROOT / "docker-compose.yml"
GAME_APP = ROOT / "game_app.py"
NGINX_CONF = ROOT / "nginx.conf"
REQUIREMENTS = ROOT / "requirements.txt"
LEGACY_MODULES = [
    ROOT / "lyrics_alignment_experiment.py",
    ROOT / "question_logic.py",
    ROOT / "yandex_request_logic.py",
    ROOT / "yandex_url_logic.py",
]


class RuntimeLayoutTests(unittest.TestCase):
    def test_generator_service_contains_local_runtime_logic(self):
        source = GENERATOR_SERVICE.read_text(encoding="utf-8")

        self.assertNotIn("import question_logic as qlogic", source)
        self.assertNotIn("from yandex_url_logic import dedupe_yandex_track_urls", source)
        self.assertNotIn("from yandex_request_logic import YandexRequestThrottle", source)

        self.assertIn("class YandexRequestThrottle:", source)
        self.assertIn("def dedupe_yandex_track_urls(", source)
        self.assertIn("def should_keep_word_token(", source)
        self.assertIn("def recover_segment_skipped_tokens(", source)
        self.assertIn("def question_window_has_enough_context(", source)
        self.assertIn("def rank_lyrics_candidates(", source)
        self.assertIn("def remap_words_to_official_by_line(", source)
        self.assertIn("from types import SimpleNamespace", source)
        self.assertIn("qlogic = SimpleNamespace(", source)
        self.assertNotIn("from lyrics_alignment_experiment import remap_words_to_official_by_line", source)

    def test_generator_service_does_not_keep_legacy_llm_or_lrc_dead_paths(self):
        source = GENERATOR_SERVICE.read_text(encoding="utf-8")

        self.assertNotIn("import ollama", source)
        self.assertNotIn("ollama_client =", source)
        self.assertNotIn("LLM_MODEL =", source)
        self.assertNotIn("LLM_NUM_PREDICT =", source)
        self.assertNotIn("LLM_TEMPERATURE =", source)
        self.assertNotIn("OLLAMA_HOST =", source)

        self.assertNotIn("def correct_transcription_with_llm(", source)
        self.assertNotIn("def parse_simple_response(", source)
        self.assertNotIn("def get_quiz_data_llm(", source)
        self.assertNotIn("def validate_llm_response(", source)
        self.assertNotIn("def find_safest_occurrence_index(", source)
        self.assertNotIn("def select_word_from_lrc(", source)
        self.assertNotIn("def calculate_timings_from_lrc(", source)
        self.assertNotIn("def forced_align_lyrics(", source)
        self.assertNotIn("def preprocess_for_alignment(", source)

    def test_legacy_helper_modules_are_removed_after_runtime_merge(self):
        for legacy_module in LEGACY_MODULES:
            with self.subTest(module=legacy_module.name):
                self.assertFalse(legacy_module.exists(), f"Legacy module still present: {legacy_module.name}")

    def test_game_app_uses_non_eventlet_socketio_runtime(self):
        source = GAME_APP.read_text(encoding="utf-8")

        self.assertNotIn("import eventlet", source)
        self.assertNotIn("eventlet.monkey_patch()", source)
        self.assertIn("async_mode='threading'", source)

    def test_game_app_uses_gthread_in_compose(self):
        compose = DOCKER_COMPOSE.read_text(encoding="utf-8")

        self.assertIn("command: gunicorn --worker-class gthread", compose)
        self.assertIn("--threads 100", compose)
        self.assertNotIn("command: gunicorn -k eventlet", compose)
        self.assertNotIn("command: python game_app.py", compose)

    def test_requirements_replace_eventlet_with_simple_websocket(self):
        requirements = REQUIREMENTS.read_text(encoding="utf-8")

        self.assertIn("flask-socketio>=5.3,<6", requirements)
        self.assertIn("python-socketio>=5.11,<6", requirements)
        self.assertIn("python-engineio>=4.8,<5", requirements)
        self.assertIn("simple-websocket>=1.0,<2", requirements)
        self.assertNotIn("\neventlet\n", f"\n{requirements}\n")

    def test_cpu_compose_uses_cpu_friendly_runtime_tuning(self):
        compose = DOCKER_COMPOSE.read_text(encoding="utf-8")

        self.assertIn("- WHISPER_THREADS=8", compose)
        self.assertIn("- WHISPER_BATCH_SIZE=8", compose)
        self.assertIn("- MDX_BATCH_SIZE=1", compose)

    def test_nginx_uses_docker_dns_reresolution_for_game_app(self):
        nginx_conf = NGINX_CONF.read_text(encoding="utf-8")

        self.assertIn("map $http_upgrade $connection_upgrade {", nginx_conf)
        self.assertIn("'' close;", nginx_conf)
        self.assertIn("resolver 127.0.0.11 ipv6=off valid=10s;", nginx_conf)
        self.assertIn("set $game_app_http http://game-app:5000;", nginx_conf)
        self.assertIn("set $game_app_socketio http://game-app:5000;", nginx_conf)
        self.assertIn("proxy_pass $game_app_http;", nginx_conf)
        self.assertIn("proxy_pass $game_app_socketio$request_uri;", nginx_conf)
        self.assertIn("proxy_set_header Connection $connection_upgrade;", nginx_conf)


if __name__ == "__main__":
    unittest.main()
