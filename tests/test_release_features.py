import unittest
from pathlib import Path

from tests.game_app_logic_loader import load_game_app_logic
from tests.generator_service_logic_loader import load_generator_service_logic


generator_logic = load_generator_service_logic()
game_app_logic = load_game_app_logic()

is_separator_fragment_candidate = generator_logic["is_separator_fragment_candidate"]
get_algorithmic_choice = generator_logic["get_algorithmic_choice"]
build_question_payload = generator_logic["build_question_payload"]
validate_player_answer_text = game_app_logic["validate_player_answer_text"]
MIN_PLAYER_ANSWER_LENGTH = game_app_logic["MIN_PLAYER_ANSWER_LENGTH"]

BASE_DIR = Path(__file__).resolve().parents[1]
INDEX_TEMPLATE = BASE_DIR / "src" / "game" / "templates" / "index.html"


def make_word(word, start, end, **extra):
    payload = {
        "word": word,
        "start": start,
        "end": end,
        "confidence": 0.9,
        "is_eol": extra.pop("is_eol", False),
    }
    payload.update(extra)
    return payload


class ReleaseFeatureTests(unittest.TestCase):
    def test_separator_fragment_is_blocked(self):
        word = make_word(
            "pirate",
            10.0,
            10.4,
            segment_text="pirate-bandit rides tonight",
        )

        self.assertTrue(is_separator_fragment_candidate(word))

    def test_algorithmic_choice_skips_separator_fragment(self):
        words = [
            make_word("pirate", 12.0, 12.4, is_eol=True, segment_text="pirate-bandit rides"),
            make_word("legend", 24.0, 24.6, is_eol=True, segment_text="the living legend"),
        ]

        self.assertEqual(get_algorithmic_choice(words), 1)

    def test_build_question_payload_adds_track_meta_only_for_whisper_only_questions(self):
        whisper_payload = build_question_payload(
            question_id="abc123",
            context_str="Write it one more ___",
            answer_text="time",
            track_meta="Artist - Title",
            has_lyrics=False,
        )
        lyrics_payload = build_question_payload(
            question_id="def456",
            context_str="Write it one more ___",
            answer_text="time",
            track_meta="Artist - Title",
            has_lyrics=True,
        )

        self.assertEqual(whisper_payload["track_meta"], "Artist - Title")
        self.assertEqual(lyrics_payload["track_meta"], "")

    def test_player_answer_validation_rejects_short_answers(self):
        is_valid, message = validate_player_answer_text("abc")

        self.assertFalse(is_valid)
        self.assertIn(str(MIN_PLAYER_ANSWER_LENGTH), message)

    def test_player_template_contains_meta_and_validation_hooks(self):
        template = INDEX_TEMPLATE.read_text(encoding="utf-8")

        self.assertIn('id="questionTrackMetaDisplay"', template)
        self.assertIn('id="resultTrackMetaDisplay"', template)
        self.assertIn("renderTrackMeta(", template)
        self.assertIn('id="answerError"', template)
        self.assertIn("answer_validation_error", template)


if __name__ == "__main__":
    unittest.main()
