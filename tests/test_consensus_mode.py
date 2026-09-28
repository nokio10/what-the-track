import importlib.util
import itertools
import unittest
from pathlib import Path

from tests.generator_service_logic_loader import load_generator_service_logic


logic = load_generator_service_logic()
consensus_clusters = logic["consensus_clusters"]
consensus_tokens = logic["consensus_tokens"]
consensus_word_tiers = logic["consensus_word_tiers"]
map_tiers_to_words = logic["map_tiers_to_words"]
mix_support = logic["mix_support"]
select_general_question = logic["select_general_question"]

ROOT = Path(__file__).resolve().parents[1]
BASE = ["первый", "второй", "третий", "четвертый", "пятый", "шестой", "седьмой"]


def words(tokens, step=0.5, start=20.0, shift=0.0):
    return [{"word": t, "start": start + i * step + shift,
             "end": start + i * step + shift + 0.4} for i, t in enumerate(tokens)]


def signature(clusters, token_lists):
    """Кластеры без имён моделей: слово, цепочка и набор таймкодов участников."""
    return sorted(
        (c["token"], c["run"],
         tuple(sorted(round(token_lists[name][pos][2], 3) for name, pos in c["members"].items())))
        for c in clusters)


class ConsensusSymmetryTests(unittest.TestCase):
    def inputs(self):
        noisy = list(BASE)
        noisy[3] = "иной"
        extra = BASE[:2] + ["лишнее"] + BASE[2:]
        return [words(BASE), words(noisy, shift=0.1), words(extra, shift=-0.1)]

    def test_permuting_models_does_not_change_result(self):
        data = self.inputs()
        reference = None
        for perm in itertools.permutations(data):
            token_lists = {name: consensus_tokens(w) for name, w in zip("abc", perm)}
            got = signature(consensus_clusters(token_lists), token_lists)
            if reference is None:
                reference = got
            self.assertEqual(got, reference)

    def test_word_missed_by_one_model_is_not_confirmed(self):
        noisy = list(BASE)
        noisy[3] = "иной"
        token_lists = {"a": consensus_tokens(words(BASE)), "b": consensus_tokens(words(BASE)),
                       "c": consensus_tokens(words(noisy))}
        self.assertNotIn("четвертый", [c["token"] for c in consensus_clusters(token_lists)])

    def test_same_word_far_apart_in_time_is_not_matched(self):
        token_lists = {"a": consensus_tokens(words(BASE)),
                       "b": consensus_tokens(words(BASE, shift=3.0))}
        self.assertEqual(consensus_clusters(token_lists), [])


class ConsensusTierTests(unittest.TestCase):
    def test_tiers_follow_confirmed_run(self):
        tiers = consensus_word_tiers({"w": words(BASE), "c": words(BASE), "r": words(BASE)}, "w")
        self.assertEqual(tiers, {3: "B", 4: "B", 5: "A", 6: "A"})

    def test_extra_word_in_any_model_breaks_the_run(self):
        extra = BASE[:2] + ["лишнее"] + BASE[2:]
        tiers = consensus_word_tiers({"w": words(BASE), "c": words(BASE), "r": words(extra)}, "w")
        self.assertEqual(tiers, {5: "B", 6: "B"})

    def test_long_pause_in_context_rejects(self):
        tiers = consensus_word_tiers({"w": words(BASE, step=3.0), "c": words(BASE, step=3.0)}, "w")
        self.assertEqual(tiers, {})

    def test_tiers_are_mapped_from_raw_to_filtered_words(self):
        raw = words(BASE)
        filtered = [w for i, w in enumerate(raw) if i != 1]
        self.assertEqual(map_tiers_to_words({5: "A", 6: "A"}, raw, filtered), {4: "A", 5: "A"})


class LineEndTests(unittest.TestCase):
    def test_pause_after_and_none_before_marks_line_end(self):
        line = words(BASE[:4]) + words(BASE[4:], start=23.0)
        info = logic["consensus_word_info"]({"w": line, "c": line, "r": line}, "w")
        # «четвертый» стоит перед паузой 1.1 с — конец строки; «пятый» после
        # паузы — начало следующей, и концом строки быть не может.
        self.assertTrue(logic["consensus_line_end"]({"pause_after": 1.1, "pause_before": 0.1}))
        self.assertFalse(logic["consensus_line_end"]({"pause_after": 0.1, "pause_before": 1.1}))
        self.assertFalse(logic["consensus_line_end"]({"pause_after": 1.1, "pause_before": 1.1}))
        self.assertEqual(info[3], ("B", 1, False))

    def test_priority_strong_end_then_weak_end_then_rest(self):
        info = {1: ("A", 0, False), 2: ("B", 1, False), 3: ("A", 2, False)}
        order = logic["consensus_priority"](info)
        rank = lambda idx: next(k for k, group in enumerate(order) if idx in group)
        self.assertLess(rank(3), rank(2))
        self.assertLess(rank(2), rank(1))

    def test_nouns_go_first_within_same_phrase_end(self):
        info = {1: ("A", 2, False), 2: ("B", 2, False), 3: ("A", 1, False)}
        pos = {1: "content", 2: "noun", 3: "noun"}
        order = logic["consensus_priority"](info, pos)
        rank = lambda idx: next(k for k, group in enumerate(order) if idx in group)
        # Существительное с сильным концом раньше глагола с сильным концом, а тот
        # раньше существительного со слабым: конец фразы важнее части речи.
        self.assertLess(rank(2), rank(1))
        self.assertLess(rank(1), rank(3))

    @unittest.skipUnless(importlib.util.find_spec("pymorphy3"), "pymorphy3 не установлен")
    def test_function_words_are_not_allowed(self):
        words_ = [{"word": w} for w in ("чтобы", "такая", "дорога", "промахнулся")]
        info = {i: ("A", 1, False) for i in range(4)}
        allowed, _order = logic["consensus_candidates"](info, words_)
        self.assertEqual(allowed, {2, 3})
        self.assertEqual(logic["answer_pos_class"]("дорога"), "noun")


class PunctuationTests(unittest.TestCase):
    def test_strong_and_weak_phrase_end(self):
        signals = logic["consensus_phrase_signals"]
        text = [{"word": "шли"}, {"word": "домой."}, {"word": "вдоль"}, {"word": "реки,"}]
        self.assertEqual(signals(text, 1), (2, False))
        self.assertEqual(signals(text, 3), (1, False))
        self.assertEqual(signals(text, 0), (0, False))

    def test_capital_or_word_after_punctuation_starts_phrase(self):
        signals = logic["consensus_phrase_signals"]
        text = [{"word": "домой."}, {"word": "вдоль"}, {"word": "Река"}, {"word": "течёт"}]
        self.assertTrue(signals(text, 1)[1])     # сразу после точки
        self.assertTrue(signals(text, 2)[1])     # с заглавной буквы
        self.assertFalse(signals(text, 3)[1])

    def test_punctuation_model_sets_levels_and_starts(self):
        plain = words(BASE)
        punct = [dict(w) for w in plain]
        punct[4]["word"] = "пятый."
        punct[5]["word"] = "Шестой"
        info = logic["consensus_word_info"]({"w": plain, "c": plain, "r": punct}, "w",
                                            punctuation_model="r")
        self.assertEqual(info[4][1], 2)          # точка у GigaAM — сильный конец
        self.assertTrue(info[5][2])              # после точки и с заглавной — начало

    @unittest.skipUnless(importlib.util.find_spec("pymorphy3"), "pymorphy3 не установлен")
    def test_phrase_start_is_never_an_answer(self):
        info = {1: ("A", 2, False), 2: ("A", 2, True)}
        allowed, _order = logic["consensus_candidates"](
            info, [{"word": "а"}, {"word": "дорога"}, {"word": "Дорога"}])
        self.assertIn(1, allowed)
        self.assertNotIn(2, allowed)


class MixSupportTests(unittest.TestCase):
    def test_counts_models_hearing_the_word_nearby(self):
        target = {"word": "Пятый,", "start": 22.0, "end": 22.4}
        near = [{"word": "пятый", "start": 22.3, "end": 22.7}]
        far = [{"word": "пятый", "start": 23.0, "end": 23.4}]
        other = [{"word": "шестой", "start": 22.0, "end": 22.4}]
        self.assertEqual(mix_support(target, [near, far, other]), 1)
        self.assertEqual(mix_support(target, [near, near, far]), 2)


class SelectGeneralQuestionTests(unittest.TestCase):
    def song(self):
        tokens = ["слово%02d" % i for i in range(40)]
        return [{"word": t.replace("слово", "песня"), "start": 13.0 + i * 1.0,
                 "end": 13.0 + i * 1.0 + 0.6, "is_eol": i % 4 == 3}
                for i, t in enumerate(tokens)]

    def test_empty_allowed_set_gives_no_question(self):
        self.assertIsNone(select_general_question(self.song(), allowed=set()))

    def test_extended_range_is_reached_when_normal_range_is_empty(self):
        # Индекс 31 из 40 (позиция 0.78) есть только в расширенном окне
        # 0.20–0.85. Раньше функция сдавалась, не дойдя до него.
        song = self.song()
        self.assertEqual(select_general_question(song, allowed={31}), 31)

    def test_answer_is_taken_only_from_allowed(self):
        song = self.song()
        free = select_general_question(song)
        self.assertIsNotNone(free)
        allowed = {i for i in range(len(song)) if i != free}
        picked = select_general_question(song, allowed=allowed)
        self.assertIn(picked, allowed)


class RawSpoilerTests(unittest.TestCase):
    def test_answer_heard_by_any_model_inside_question_window(self):
        target = {"word": "дорога", "start": 60.0, "end": 60.5}
        index = logic["raw_word_index"]([[{"word": "Дорога,", "start": 45.0, "end": 45.4}]])
        self.assertTrue(logic["raw_spoiler"](target, index))

    def test_window_is_the_actual_question_clip(self):
        # Клип начинается с начала слова и бывает длиннее 28 с: повтор за 30 с до
        # ответа слышен, если клип начался раньше.
        target = {"word": "дорога", "start": 60.0, "end": 60.5}
        index = logic["raw_word_index"]([[{"word": "дорога", "start": 29.8, "end": 30.2}]])
        self.assertFalse(logic["raw_spoiler"](target, index))
        self.assertTrue(logic["raw_spoiler"](target, index, q_start_sec=29.0))

    def test_same_occurrence_and_far_occurrence_are_not_spoilers(self):
        target = {"word": "дорога", "start": 60.0, "end": 60.5}
        index = logic["raw_word_index"]([
            [{"word": "дорога", "start": 60.3, "end": 60.8}],   # тот же ответ у другой модели
            [{"word": "дорога", "start": 20.0, "end": 20.4}],   # раньше окна вопроса
        ])
        self.assertFalse(logic["raw_spoiler"](target, index))


class LegacyPathTests(unittest.TestCase):
    def test_empty_normal_window_falls_back_to_extended(self):
        # Годные ответы только в расширенном окне позиции: раньше старый путь без
        # текста пропускал такой трек, хотя select_general_question его брал.
        names = ["небосвод", "морской", "ветрило", "солнышко", "городок", "лодочка",
                 "песочек", "каменный", "птичий", "звёздный", "речной", "горный"]
        words = []
        for i in range(40):
            start = 13.0 + i
            word = "ой" if 12 <= i <= 28 else names[i % len(names)] + "ка" * (i // 12)
            words.append({"word": word, "start": start, "end": start + 0.6, "is_eol": i % 4 == 3})
        self.assertEqual(logic["score_candidates"](words, extended_range=False), [])
        picked = logic["get_algorithmic_choice"](words)
        self.assertIsNotNone(picked)
        self.assertEqual(picked, logic["select_general_question"](words))


class GenerationSubprocessTests(unittest.TestCase):
    def test_child_status_changes_are_forwarded_in_order(self):
        import queue
        events = queue.Queue()
        status = logic["_ForwardedStatus"](events)
        status["logs"] = []            # generation_task начинает с чистого лога
        status["logs"].append("[ 1.0s] трек 1")
        status["progress"] = 50
        status["status"] = "finished"
        got = []
        while not events.empty():
            got.append(events.get_nowait())
        self.assertEqual(got, [("reset_logs",), ("log", "[ 1.0s] трек 1"),
                               ("set", "progress", 50), ("set", "status", "finished")])
        self.assertEqual(status["logs"], ["[ 1.0s] трек 1"])

    def test_parent_applies_events_to_job_status(self):
        apply = logic["_apply_generation_event"]
        job_status = logic["job_status"]
        job_status.update(is_busy=True, progress=0, logs=["старое"], status="running")
        for event in [("reset_logs",), ("log", "a"), ("set", "progress", 100),
                      ("set", "status", "finished"), ("gigaam_failed_at", 123.0)]:
            apply(event)
        self.assertEqual(job_status["logs"], ["a"])
        self.assertEqual((job_status["progress"], job_status["status"]), (100, "finished"))
        self.assertEqual(logic["_GIGAAM_STATE"]["failed_at"], 123.0)

    def test_worker_runs_generation_in_subprocess_by_default(self):
        source = (ROOT / "src" / "game" / "generator_service.py").read_text(encoding="utf-8")
        self.assertIn('os.environ.get("GENERATION_IN_SUBPROCESS", "1")', source)
        self.assertIn("run_generation_isolated(task_data)", source)
        self.assertIn('if multiprocessing.current_process().name == "MainProcess":', source)


class ComputeTypeTests(unittest.TestCase):
    def test_float16_is_never_used_on_cpu(self):
        compute_type = logic["whisper_compute_type"]
        saved = compute_type.__globals__["COMPUTE_TYPE"]
        try:
            compute_type.__globals__["COMPUTE_TYPE"] = ""
            self.assertEqual(compute_type("cuda"), "float16")
            self.assertEqual(compute_type("cpu"), "int8")
            compute_type.__globals__["COMPUTE_TYPE"] = "float16"   # compose с GPU без CUDA
            self.assertEqual(compute_type("cpu"), "int8")
            self.assertEqual(compute_type("cuda"), "float16")
            compute_type.__globals__["COMPUTE_TYPE"] = "int8_float32"
            self.assertEqual(compute_type("cpu"), "int8_float32")
        finally:
            compute_type.__globals__["COMPUTE_TYPE"] = saved


class GeneratorInputTests(unittest.TestCase):
    def test_game_id_is_restricted_to_safe_characters(self):
        is_valid_game_id = logic["is_valid_game_id"]
        self.assertTrue(is_valid_game_id("ABCD"))
        self.assertFalse(is_valid_game_id("../etc"))
        self.assertFalse(is_valid_game_id(12))

    def test_env_float_falls_back_on_empty_or_garbage(self):
        import os
        env_float = logic["_env_float"]
        for value, expected in (("", 5.0), ("12.5", 12.5), ("3m", 5.0)):
            os.environ["MUSIC_GAME_TEST_FLOAT"] = value
            try:
                self.assertEqual(env_float("MUSIC_GAME_TEST_FLOAT", 5.0), expected)
            finally:
                del os.environ["MUSIC_GAME_TEST_FLOAT"]


class ExpandYandexUrlsTests(unittest.TestCase):
    class _Track:
        def __init__(self, track_id):
            self.id = track_id

    class _Short:
        def __init__(self, track_id, rich=True):
            self.id = track_id
            self.track = ExpandYandexUrlsTests._Track(track_id) if rich else None

    class _Client:
        def albums_with_tracks(self, album_id):
            track = ExpandYandexUrlsTests._Track
            return type("Album", (), {"volumes": [[track("1"), track("2")], [track("3")]]})()

        def users_playlists(self, kind, user):
            short = ExpandYandexUrlsTests._Short
            return type("Playlist", (), {"tracks": [short("7"), short("8", rich=False)]})()

    def test_album_and_playlist_links_become_track_links(self):
        messages = []
        urls = logic["expand_yandex_urls"](
            ["https://music.yandex.ru/album/42/track/5",
             "https://music.yandex.ru/album/42",
             "https://music.yandex.ru/users/someone/playlists/3",
             "https://example.com/other"],
            self._Client(), logger=messages.append)
        self.assertEqual(urls, [
            "https://music.yandex.ru/album/42/track/5",
            "https://music.yandex.ru/album/42/track/1",
            "https://music.yandex.ru/album/42/track/2",
            "https://music.yandex.ru/album/42/track/3",
            "https://music.yandex.ru/track/7",
            "https://music.yandex.ru/track/8",
            "https://example.com/other",
        ])
        self.assertEqual(len(messages), 2)

    def test_failed_expansion_is_logged_and_skipped(self):
        class Broken:
            def albums_with_tracks(self, album_id):
                raise RuntimeError("нет доступа")
        messages = []
        urls = logic["expand_yandex_urls"](["https://music.yandex.ru/album/9"], Broken(),
                                           logger=messages.append)
        self.assertEqual(urls, [])
        self.assertIn("нет доступа", messages[0])


class ConsensusModeDefaultTests(unittest.TestCase):
    def test_mode_is_on_by_default_and_gigaam_always_starts(self):
        source = (ROOT / "src" / "game" / "generator_service.py").read_text(encoding="utf-8")
        self.assertIn('CONSENSUS_MODE = os.environ.get("CONSENSUS_MODE", "on")', source)
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn("CONSENSUS_MODE=${CONSENSUS_MODE:-on}", compose)
        self.assertNotIn("profiles:", compose)

    def test_yandex_token_can_come_from_env(self):
        source = (ROOT / "src" / "game" / "generator_service.py").read_text(encoding="utf-8")
        self.assertIn('YANDEX_TOKEN = os.environ.get("YANDEX_TOKEN", "").strip()', source)
        self.assertIn("if not token and YANDEX_TOKEN:", source)
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn("YANDEX_TOKEN=${YANDEX_TOKEN:-}", compose)

    def test_no_secret_is_written_into_compose_files(self):
        for name in ("docker-compose.yml", "docker-compose.nvidia.yml"):
            self.assertNotRegex((ROOT / name).read_text(encoding="utf-8"), r"HF_TOKEN=hf_")

    @unittest.skipUnless(importlib.util.find_spec("requests"), "requests не установлен")
    def test_track_failure_is_told_apart_from_service_failure(self):
        import requests

        class Response:
            def __init__(self, status, payload):
                self.status_code, self._payload, self.text = status, payload, str(payload)

            def json(self):
                return self._payload

        transcribe = logic["transcribe_gigaam"]
        saved = requests.post
        try:
            requests.post = lambda *a, **k: Response(500, {"error": "RuntimeError: Failed to load audio"})
            with self.assertRaises(logic["GigaamTrackError"]) as caught:
                transcribe("/app/media/a.wav", "v3_ctc")
            self.assertIn("Failed to load audio", str(caught.exception))
            requests.post = lambda *a, **k: Response(503, {"error": "нужен HF_TOKEN"})
            with self.assertRaises(logic["GigaamUnavailable"]) as caught:
                transcribe("/app/media/a.wav", "v3_ctc")
            self.assertNotIsInstance(caught.exception, logic["GigaamTrackError"])
        finally:
            requests.post = saved

    @unittest.skipUnless(importlib.util.find_spec("requests"), "requests не установлен")
    def test_unreachable_gigaam_raises_its_own_error(self):
        transcribe = logic["transcribe_gigaam"]
        transcribe.__globals__["GIGAAM_URL"] = "http://127.0.0.1:9"
        with self.assertRaises(logic["GigaamUnavailable"]):
            transcribe("/app/media/нет.mp3", "v3_ctc")


if __name__ == "__main__":
    unittest.main()
