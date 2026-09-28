import unittest

from tests.generator_service_logic_loader import load_generator_service_logic


logic = load_generator_service_logic()
QUESTION_WORD_GUARD_MS = logic["QUESTION_WORD_GUARD_MS"]
build_context_string = logic["build_context_string"]
calculate_timings = logic["calculate_timings"]
question_window_has_enough_context = logic["question_window_has_enough_context"]
recover_segment_skipped_tokens = logic["recover_segment_skipped_tokens"]
get_algorithmic_choice = logic["get_algorithmic_choice"]
select_word_from_lyrics_algorithmically = logic["select_word_from_lyrics_algorithmically"]
should_keep_word_token = logic["should_keep_word_token"]
is_separator_fragment_candidate = logic["is_separator_fragment_candidate"]
build_question_payload = logic["build_question_payload"]


def make_word(
    word,
    start,
    end,
    *,
    is_eol=False,
    lyrics_line_idx=None,
    lyrics_line="",
    confidence=0.9,
    skipped_before=None,
    segment_text="",
):
    payload = {
        "word": word,
        "start": start,
        "end": end,
        "is_eol": is_eol,
        "confidence": confidence,
    }
    if lyrics_line_idx is not None:
        payload["lyrics_line_idx"] = lyrics_line_idx
    if lyrics_line:
        payload["lyrics_line"] = lyrics_line
    if skipped_before:
        payload["skipped_before"] = skipped_before
    if segment_text:
        payload["segment_text"] = segment_text
    return payload


class QuestionLogicTests(unittest.TestCase):
    def test_keeps_common_single_letter_russian_words(self):
        for word in ("и", "в", "к", "с", "я", "а", "о", "у"):
            with self.subTest(word=word):
                self.assertTrue(should_keep_word_token(word, confidence=0.35))

    def test_rejects_unknown_single_letter_noise(self):
        for word in ("z", "q", "1", "-", ""):
            with self.subTest(word=word):
                self.assertFalse(should_keep_word_token(word, confidence=0.95))

    def test_calculate_timings_uses_100ms_guard_before_answer_word(self):
        words = [
            make_word("медленно", 10.00, 10.60),
            make_word("гаснет", 35.00, 35.50, is_eol=True),
        ]

        question_times, answer_times = calculate_timings(words, 1)

        self.assertEqual(question_times[1], 35000 - QUESTION_WORD_GUARD_MS)
        self.assertLess(question_times[1], int(words[1]["start"] * 1000))
        self.assertEqual(answer_times[0], 30000)

    def test_algorithmic_choice_is_deterministic_and_avoids_repeated_word(self):
        words = [
            make_word("вступление", 5.0, 5.7),
            make_word("белый", 13.0, 13.4),
            make_word("снег", 13.5, 14.0, is_eol=True),
            make_word("белый", 18.0, 18.4),
            make_word("снег", 18.5, 19.0, is_eol=True),
            make_word("неизбежность", 24.0, 24.8, is_eol=True),
            make_word("тает", 30.0, 30.4),
            make_word("город", 30.5, 31.1, is_eol=True),
        ]

        self.assertEqual(get_algorithmic_choice(words), 5)
        self.assertEqual(get_algorithmic_choice(words), 5)

    def test_lyrics_selection_prefers_occurrence_with_matching_line_mapping(self):
        official_lyrics = "\n".join(
            [
                "Город засыпает во тьме",
                "Остается свет в окне",
                "И внезапно слышу я",
                "Как приходит тишина",
                "Чтобы снова встретить рассвет",
                "И оставить в небе след",
            ]
        )
        # Одно и то же слово-рифма дважды: один раз на своей строке, второй —
        # ошибочно привязанное к чужой. Слова идут плотно: _score_occurrence
        # отбрасывает слово после паузы > 5 с, и в прежнем варианте теста
        # паузы по 11 с проверяли уже это правило, а не привязку к строке.
        words = [
            make_word("тьме", 13.0, 13.4, is_eol=True, lyrics_line_idx=0, lyrics_line="Город засыпает во тьме"),
            make_word("след", 16.0, 16.5, is_eol=True, lyrics_line_idx=5, lyrics_line="И оставить в небе след"),
            make_word("след", 19.0, 19.5, is_eol=True, lyrics_line_idx=1, lyrics_line="Остается свет в окне"),
        ]

        target_idx, answer_line, answer_line_idx = select_word_from_lyrics_algorithmically(official_lyrics, words)

        self.assertEqual(target_idx, 1)
        self.assertEqual(answer_line, "И оставить в небе след")
        self.assertEqual(answer_line_idx, 5)

    def test_build_context_string_uses_skipped_words_across_whole_context(self):
        words = [
            make_word("Замедляю", 98.00, 98.40),
            make_word("и", 98.45, 98.55),
            make_word("растягиваю", 98.60, 99.20),
            make_word("вечер", 99.25, 99.70),
            make_word("пою", 100.00, 100.30, skipped_before=["ведь", "я"]),
            make_word("только", 100.35, 100.70),
            make_word("редкие", 100.75, 101.20, skipped_before=["в"]),
            make_word("минуты", 101.30, 101.75),
        ]

        context = build_context_string(words, 7)

        self.assertEqual(
            context,
            "Замедляю и растягиваю вечер ведь я пою только в редкие ___",
        )

    def test_recover_segment_skipped_tokens_restores_short_preposition(self):
        segment_words = [
            make_word("Неужели", 88.80, 90.02),
            make_word("так", 90.08, 90.30),
            make_word("трудно", 90.36, 91.66),
            make_word("театр", 91.74, 92.26),
            make_word("успевать", 92.38, 93.98),
        ]

        restored = recover_segment_skipped_tokens(
            "Неужели так трудно в театр успевать",
            segment_words,
        )

        self.assertEqual(restored[3].get("skipped_before"), ["в"])

    def test_recover_segment_skipped_tokens_restores_short_meaningful_word(self):
        segment_words = [
            make_word("пустые", 91.02, 91.66),
            make_word("споры", 91.72, 92.44),
            make_word("год", 92.64, 92.88),
            make_word("ожидании", 92.98, 93.80),
        ]

        restored = recover_segment_skipped_tokens(
            "пустые споры весь год в ожидании",
            segment_words,
        )

        self.assertEqual(restored[2].get("skipped_before"), ["весь"])

    def test_build_context_string_deduplicates_overlapping_skipped_sequence(self):
        words = [
            make_word("Жди", 166.00, 166.20),
            make_word("пару", 166.22, 166.45),
            make_word("секунд", 166.47, 166.67),
            make_word("жди", 166.69, 166.88),
            make_word("нас", 166.90, 167.05),
            make_word("просто", 167.07, 167.27),
            make_word("жди", 167.29, 167.45),
            make_word("Пару", 167.47, 167.67, skipped_before=["Жди", "нас", "просто", "жди"]),
            make_word("секунд", 167.69, 167.97),
            make_word("до", 168.01, 168.21),
            make_word("рассвета", 168.27, 168.67),
            make_word("слушай", 168.73, 169.11),
            make_word("не", 169.15, 169.27),
            make_word("засыпай", 169.29, 169.91),
        ]

        context = build_context_string(words, 13)

        self.assertEqual(
            context,
            "Жди пару секунд жди нас просто жди Пару секунд до рассвета слушай не ___",
        )

    def test_rejects_question_window_with_long_intro_and_too_few_words(self):
        words = [
            make_word("Что", 179.18, 179.53),
            make_word("что", 179.59, 179.92),
            make_word("еще", 179.98, 180.28),
            make_word("произошло", 180.34, 181.20),
        ]

        is_ok, reason = question_window_has_enough_context(
            words,
            target_idx=3,
            q_start_ms=152261,
            q_end_ms=180261,
        )

        self.assertFalse(is_ok)
        self.assertEqual(reason, "leading_instrumental")

    def test_build_context_string_appends_only_missing_tail_from_repeated_target_skipped_block(self):
        words = [
            make_word("нас", 165.00, 165.20),
            make_word("пару", 165.22, 165.45),
            make_word("секунд", 165.47, 165.70),
            make_word("жди", 165.74, 165.92),
            make_word("нас", 165.96, 166.15),
            make_word("просто", 166.19, 166.42),
            make_word("жди", 166.46, 166.62),
            make_word("пару", 166.66, 166.92),
            make_word("секунд", 166.96, 167.18),
            make_word("до", 167.22, 167.36),
            make_word("рассвета", 167.40, 167.82),
            make_word("слушай", 167.86, 168.14),
            make_word(
                "засыпай",
                168.29,
                168.91,
                skipped_before=[
                    "Жди",
                    "нас",
                    "пару",
                    "секунд",
                    "жди",
                    "нас",
                    "Просто",
                    "жди",
                    "пару",
                    "секунд",
                    "до",
                    "рассвета",
                    "слушай",
                    "не",
                ],
            ),
        ]

        context = build_context_string(words, 12)

        self.assertEqual(
            context,
            "нас пару секунд жди нас просто жди пару секунд до рассвета слушай не ___",
        )

    def test_build_context_string_prefers_clean_answer_line_in_lyrics_mode(self):
        words = [
            make_word("нас", 165.00, 165.20),
            make_word("пару", 165.22, 165.45),
            make_word("секунд", 165.47, 165.70),
            make_word("жди", 165.74, 165.92),
            make_word("просто", 166.19, 166.42),
            make_word("до", 167.22, 167.36),
            make_word("рассвета", 167.40, 167.82),
            make_word("слушай", 167.86, 168.14),
            make_word(
                "засыпай",
                168.29,
                168.91,
                skipped_before=[
                    "Жди",
                    "нас",
                    "пару",
                    "секунд",
                    "жди",
                    "пару",
                    "секунд",
                    "до",
                    "рассвета",
                    "слушай",
                    "не",
                ],
            ),
        ]

        context = build_context_string(
            words,
            8,
            is_lyrics_mode=True,
            answer_line="Жди пару секунд, жди нас просто жди пару секунд до рассвета, слушай, не засыпай",
        )

        self.assertEqual(
            context,
            "Жди пару секунд, жди нас просто жди пару секунд до рассвета, слушай, не ___",
        )

    def test_build_context_string_uses_two_lyrics_lines_when_available(self):
        words = [
            make_word(
                "вдали",
                160.35,
                160.80,
                is_eol=True,
                lyrics_line_idx=10,
                lyrics_line="Поставь на повтор, ах, где-то вдали",
            ),
            make_word(
                "жди",
                166.00,
                166.20,
                lyrics_line_idx=11,
                lyrics_line="Жди пару секунд, жди нас просто жди пару секунд до рассвета, слушай, не засыпай",
            ),
            make_word(
                "засыпай",
                168.29,
                168.91,
                lyrics_line_idx=11,
                lyrics_line="Жди пару секунд, жди нас просто жди пару секунд до рассвета, слушай, не засыпай",
            ),
        ]

        context = build_context_string(
            words,
            2,
            is_lyrics_mode=True,
            answer_line="Жди пару секунд, жди нас просто жди пару секунд до рассвета, слушай, не засыпай",
        )

        self.assertEqual(
            context,
            "Поставь на повтор, ах, где-то вдали\nЖди пару секунд, жди нас просто жди пару секунд до рассвета, слушай, не ___",
        )

    def test_build_context_string_falls_back_to_official_previous_lyrics_line(self):
        words = [
            make_word("Которым", 155.00, 155.62, lyrics_line_idx=17, lyrics_line="Которым теперь плевать на любые новости"),
            make_word("теперь", 155.66, 156.06, lyrics_line_idx=17, lyrics_line="Которым теперь плевать на любые новости"),
            make_word("плевать", 156.10, 156.52, lyrics_line_idx=17, lyrics_line="Которым теперь плевать на любые новости"),
            make_word("на", 156.58, 156.76, lyrics_line_idx=17, lyrics_line="Которым теперь плевать на любые новости"),
            make_word("любые", 156.82, 157.38, lyrics_line_idx=17, lyrics_line="Которым теперь плевать на любые новости"),
            make_word("новости", 157.48, 158.62, lyrics_line_idx=17, lyrics_line="Которым теперь плевать на любые новости"),
        ]

        lyrics_text = "\n".join(
            [
                "И вот у Веры химеры и робот-пылесос",
                "Которым теперь плевать на любые новости",
            ]
        )

        context = build_context_string(
            words,
            5,
            is_lyrics_mode=True,
            answer_line="Которым теперь плевать на любые новости",
            answer_line_idx=1,
            lyrics_text=lyrics_text,
        )

        self.assertEqual(
            context,
            "И вот у Веры химеры и робот-пылесос\nКоторым теперь плевать на любые ___",
        )

    def test_build_context_string_uses_two_lines_in_general_mode_when_previous_phrase_exists(self):
        words = [
            make_word("Потерялось,", 10.00, 10.35),
            make_word("не", 10.38, 10.50),
            make_word("нашлось", 10.53, 10.95, is_eol=True),
            make_word("Позвони", 11.15, 11.55),
            make_word("еще", 11.58, 11.78),
            make_word("разок,", 11.81, 12.20),
            make_word("ты", 12.23, 12.40),
            make_word("слышишь", 12.44, 12.96),
        ]

        context = build_context_string(words, 7)

        self.assertEqual(
            context,
            "Потерялось, не нашлось\nПозвони еще разок, ты ___",
        )

    def test_build_context_string_uses_two_lines_for_short_current_phrase(self):
        words = [
            make_word("Не", 20.00, 20.18),
            make_word("грусти.", 20.20, 20.72, is_eol=True),
            make_word("Всё", 21.02, 21.28),
            make_word("отлично", 21.35, 21.92),
        ]

        context = build_context_string(words, 3)

        self.assertEqual(
            context,
            "Не грусти.\nВсё ___",
        )


    def test_separator_fragment_candidate_is_rejected_for_hyphenated_source_word(self):
        word = make_word(
            "шкипер",
            12.0,
            12.4,
            lyrics_line="Ты шкипер-бандит, а я герой",
        )

        self.assertTrue(is_separator_fragment_candidate(word))

    def test_algorithmic_choice_skips_word_that_is_only_fragment_of_hyphenated_token(self):
        words = [
            make_word(
                "шкипер",
                11.0,
                11.4,
                is_eol=True,
                segment_text="Ты шкипер-бандит",
            ),
            make_word(
                "герой",
                21.0,
                21.5,
                is_eol=True,
                segment_text="Но я настоящий герой",
            ),
        ]

        self.assertEqual(get_algorithmic_choice(words), 1)

    def test_build_question_payload_adds_track_meta_for_whisper_only_questions(self):
        payload = build_question_payload(
            question_id="abc123",
            context_str="Позвони еще разок, ты ___",
            answer_text="слышишь",
            track_meta="Руки Вверх! - Крошка моя",
            has_lyrics=False,
        )

        self.assertEqual(payload["track_meta"], "Руки Вверх! - Крошка моя")

    def test_build_question_payload_omits_track_meta_when_lyrics_exist(self):
        payload = build_question_payload(
            question_id="abc123",
            context_str="Позвони еще разок, ты ___",
            answer_text="слышишь",
            track_meta="Руки Вверх! - Крошка моя",
            has_lyrics=True,
        )

        self.assertEqual(payload["track_meta"], "")


if __name__ == "__main__":
    unittest.main()
