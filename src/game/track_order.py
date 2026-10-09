"""Ordering questions from equal, consecutive player submissions."""
import random
import re
from urllib.parse import urlparse


def player_blocks(urls, player_count=0):
    if type(player_count) is not int or player_count < 0:
        raise ValueError('Количество игроков должно быть целым неотрицательным числом.')
    if not player_count:
        return {}
    tracks = [str(url).strip() for url in urls if str(url).strip()]
    if not tracks or len(tracks) % player_count:
        raise ValueError('Количество ссылок должно делиться на число игроков без остатка.')
    size = len(tracks) // player_count
    owners = {}
    for index, url in enumerate(tracks):
        parsed = urlparse(url)
        match = re.fullmatch(r'/(?:album/\d+/)?track/(\d+)/?', parsed.path)
        if parsed.scheme not in ('http', 'https') or parsed.hostname not in ('music.yandex.ru', 'music.yandex.com', 'music.yandex.by', 'music.yandex.kz', 'music.yandex.uz') or not match:
            raise ValueError('Для перемешивания по игрокам нужны ссылки на отдельные треки, не альбомы или плейлисты.')
        # A duplicate belongs to its first submitter; boundaries never shift.
        owners.setdefault(match.group(1), index // size)
    return owners


def mix_player_questions(groups, rng=None):
    """Keep every question, avoiding adjacent owners whenever possible."""
    rng = rng or random
    pools = [list(group) for group in groups]
    for pool in pools:
        rng.shuffle(pool)
    result, previous, repeats = [], None, 0
    while any(pools):
        available = [i for i, pool in enumerate(pools) if pool and i != previous]
        if not available:
            available = [i for i, pool in enumerate(pools) if pool]
        largest = max(len(pools[i]) for i in available)
        owner = rng.choice([i for i in available if len(pools[i]) == largest])
        repeats += owner == previous
        result.append(pools[owner].pop())
        previous = owner
    return result, repeats
