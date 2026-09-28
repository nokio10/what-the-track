import unittest

from tests.game_app_logic_loader import load_game_app_logic


logic = load_game_app_logic()
MIN_PLAYER_ANSWER_LENGTH = logic["MIN_PLAYER_ANSWER_LENGTH"]
validate_player_answer_text = logic["validate_player_answer_text"]


class GameAppLogicTests(unittest.TestCase):
    def test_rejects_answers_shorter_than_minimum_after_normalization(self):
        is_valid, message = validate_player_answer_text(" а-б ")

        self.assertFalse(is_valid)
        self.assertIn(str(MIN_PLAYER_ANSWER_LENGTH), message)

    def test_accepts_answers_with_four_or_more_meaningful_characters(self):
        is_valid, message = validate_player_answer_text("снег")

        self.assertTrue(is_valid)
        self.assertEqual(message, "")

    def test_rejects_punctuation_only_input(self):
        is_valid, message = validate_player_answer_text("...")

        self.assertFalse(is_valid)
        self.assertIn(str(MIN_PLAYER_ANSWER_LENGTH), message)


class GameStateHelpersTests(unittest.TestCase):
    def test_game_id_is_restricted_to_safe_characters(self):
        is_valid_game_id = logic["is_valid_game_id"]
        self.assertTrue(is_valid_game_id("ABCD"))
        self.assertFalse(is_valid_game_id("../x"))
        self.assertFalse(is_valid_game_id(""))
        self.assertFalse(is_valid_game_id(None))
        self.assertFalse(is_valid_game_id("A" * 33))

    def test_vip_is_first_player_on_line(self):
        players = {"s1": {"online": False}, "s2": {"online": True}, "s3": {}}
        self.assertEqual(logic["pick_vip_sid"](players), "s2")
        self.assertIsNone(logic["pick_vip_sid"]({"s1": {"online": False}}))

    def test_only_online_players_are_awaited(self):
        players = {"s1": {"online": False, "last_answer": None},
                   "s2": {"online": True, "last_answer": "снег"}}
        waiting = logic["online_players"](players)
        self.assertEqual(list(waiting), ["s2"])
        self.assertTrue(all(p["last_answer"] for p in waiting.values()))


if __name__ == "__main__":
    unittest.main()
