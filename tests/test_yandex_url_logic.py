import unittest

from tests.generator_service_logic_loader import load_generator_service_logic


logic = load_generator_service_logic()
dedupe_yandex_track_urls = logic["dedupe_yandex_track_urls"]
extract_yandex_track_identity = logic["extract_yandex_track_identity"]


class YandexUrlLogicTests(unittest.TestCase):
    def test_extracts_album_and_track_identity_from_track_url(self):
        identity = extract_yandex_track_identity(
            "https://music.yandex.ru/album/34928265/track/135089951?utm_source=foo"
        )

        self.assertEqual(identity, ("album_track", "34928265", "135089951"))

    def test_dedupes_same_album_track_with_different_querystrings(self):
        urls = [
            "https://music.yandex.ru/album/34928265/track/135089951",
            "https://music.yandex.ru/album/34928265/track/135089951?utm_source=foo",
            "https://music.yandex.ru/album/111/track/222",
        ]

        unique_urls, duplicates = dedupe_yandex_track_urls(urls)

        self.assertEqual(
            unique_urls,
            [
                "https://music.yandex.ru/album/34928265/track/135089951",
                "https://music.yandex.ru/album/111/track/222",
            ],
        )
        self.assertEqual(duplicates, ["album/34928265/track/135089951"])

    def test_keeps_same_track_id_from_different_albums(self):
        urls = [
            "https://music.yandex.ru/album/34928265/track/135089951",
            "https://music.yandex.ru/album/777777/track/135089951",
        ]

        unique_urls, duplicates = dedupe_yandex_track_urls(urls)

        self.assertEqual(unique_urls, urls)
        self.assertEqual(duplicates, [])

    def test_dedupes_bare_track_urls_by_track_id(self):
        urls = [
            "https://music.yandex.ru/track/135089951",
            "https://music.yandex.ru/track/135089951?from=copy",
        ]

        unique_urls, duplicates = dedupe_yandex_track_urls(urls)

        self.assertEqual(unique_urls, ["https://music.yandex.ru/track/135089951"])
        self.assertEqual(duplicates, ["track/135089951"])

    def test_skips_blank_urls(self):
        unique_urls, duplicates = dedupe_yandex_track_urls(
            ["", "   ", "https://music.yandex.ru/album/1/track/2"]
        )

        self.assertEqual(unique_urls, ["https://music.yandex.ru/album/1/track/2"])
        self.assertEqual(duplicates, [])


if __name__ == "__main__":
    unittest.main()
