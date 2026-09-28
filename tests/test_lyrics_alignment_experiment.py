import unittest

from tests.generator_service_logic_loader import load_generator_service_logic


remap_words_to_official_by_line = load_generator_service_logic()["remap_words_to_official_by_line"]


def make_word(word, start, end, line_idx, line_text, is_eol=False):
    return {
        "word": word,
        "start": start,
        "end": end,
        "lyrics_line_idx": line_idx,
        "lyrics_line": line_text,
        "is_eol": is_eol,
        "confidence": 0.9,
    }


class LyricsAlignmentExperimentTests(unittest.TestCase):
    def test_preserves_timed_line_when_official_line_is_too_different(self):
        words = [
            make_word("По", 47.95, 48.30, 0, "По собственному желанию"),
            make_word("собственному", 48.31, 49.10, 0, "По собственному желанию"),
            make_word("желанию", 49.11, 49.66, 0, "По собственному желанию", is_eol=True),
            make_word("Мой", 49.66, 49.78, 1, "Мой сотрудник долбоеб"),
            make_word("сотрудник", 49.84, 50.66, 1, "Мой сотрудник долбоеб"),
            make_word("долбоеб", 52.34, 53.52, 1, "Мой сотрудник долбоеб", is_eol=True),
        ]

        remapped_words, stats = remap_words_to_official_by_line(
            words,
            "По собственному желанию\nМой сотрудник максимально не продуктивен",
        )

        self.assertEqual(stats["matched_line_count"], 1)
        second_line_words = [w for w in remapped_words if w["lyrics_line_idx"] == 1]
        self.assertEqual([w["word"] for w in second_line_words], ["Мой", "сотрудник", "долбоеб"])
        self.assertTrue(all(w["lyrics_line"] == "Мой сотрудник долбоеб" for w in second_line_words))

    def test_remaps_line_and_preserves_missing_official_word_as_skipped_before(self):
        words = [
            make_word("И", 112.22, 112.47, 0, "И на этой площадке"),
            make_word("на", 112.53, 112.69, 0, "И на этой площадке"),
            make_word("этой", 112.71, 112.89, 0, "И на этой площадке"),
            make_word("площадке", 113.03, 113.48, 0, "И на этой площадке", is_eol=True),
        ]

        remapped_words, stats = remap_words_to_official_by_line(
            words,
            "И на этой дэнс площадке",
        )

        self.assertGreaterEqual(stats["match_pct"], 75.0)
        self.assertEqual(remapped_words[-1]["word"], "площадке")
        self.assertEqual(remapped_words[-1]["lyrics_line"], "И на этой дэнс площадке")
        self.assertEqual(remapped_words[-1].get("skipped_before"), ["дэнс"])

    def test_preserves_missing_short_words_as_skipped_before(self):
        words = [
            make_word("в", 21.00, 21.11, 0, "в дождь"),
            make_word("дождь", 21.12, 21.52, 0, "в дождь", is_eol=True),
        ]

        remapped_words, stats = remap_words_to_official_by_line(
            words,
            "И в дождь",
        )

        self.assertGreaterEqual(stats["match_pct"], 66.0)
        self.assertEqual(remapped_words[-1]["word"], "дождь")
        # Пропущенное слово цепляется к слову, ПЕРЕД которым оно пропущено:
        # build_context_string вставляет skipped_before перед словом, и
        # привязка к последнему слову переставила бы слова в тексте вопроса.
        self.assertEqual(remapped_words[0].get("skipped_before"), ["И"])
        self.assertIsNone(remapped_words[-1].get("skipped_before"))


if __name__ == "__main__":
    unittest.main()
