import os
import json
import random
import string
import shutil
import requests
import re
import glob
import hmac
import secrets
import threading
from functools import wraps
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room

app = Flask(__name__)
# Ключ подписи Flask. Сессий на куках игра не использует, поэтому без
# SECRET_KEY в окружении хватает случайного ключа на каждый запуск.
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or secrets.token_hex(32)

# Используем threading + gunicorn gthread/simple-websocket вместо eventlet.
socketio = SocketIO(
    app,
    # Default same-origin policy also checks WebSocket handshakes.
    async_mode='threading',
    ping_timeout=60,  # Увеличиваем timeout для медленных соединений
    ping_interval=25,  # Частота ping для проверки соединения
    logger=False,
    engineio_logger=False
)

# --- КОНФИГ ---
BASE_DIR = os.getcwd()
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
GENERATOR_URL = os.environ.get("GENERATOR_URL", "http://generator:5001")
MIN_PLAYER_ANSWER_LENGTH = 4
MAX_PLAYER_NAME_LENGTH = 32
# Пароль ведущего. Пустой — управлять игрой может любой, кто открыл /admin.
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "").strip()
GAME_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
# Сколько опросов подряд генератор может отвечать «idle», не взяв задачу.
GENERATOR_IDLE_POLLS_LIMIT = 8

if not os.path.exists(MEDIA_ROOT): os.makedirs(MEDIA_ROOT)

def is_valid_game_id(gid):
    """game_id подставляется в пути к файлам — только буквы, цифры, «_» и «-»."""
    return bool(gid) and isinstance(gid, str) and bool(GAME_ID_RE.fullmatch(gid))


def online_players(players):
    """Игроки, которые сейчас на связи. Ушедшие остаются в таблице, но не ждутся."""
    return {sid: p for sid, p in players.items() if p.get('online', True)}


def pick_vip_sid(players):
    """VIP — первый вошедший из тех, кто на связи: он может листать вопросы."""
    for sid, p in players.items():
        if p.get('online', True):
            return sid
    return None


def normalize_answer_strict(text):
    if not text: return ""
    text = text.lower().replace('ё', 'е')
    return re.sub(r'[^\w]', '', text)

# --- СОСТОЯНИЕ ИГРЫ ---
def validate_player_answer_text(text):
    normalized = normalize_answer_strict(text)
    if len(normalized) < MIN_PLAYER_ANSWER_LENGTH:
        return False, f'Ответ должен содержать минимум {MIN_PLAYER_ANSWER_LENGTH} символа(ов).'
    return True, ""


class GameState:
    def __init__(self):
        self.game_id = None
        self.is_active = False
        self.current_q_index = -1
        self.questions = []
        self.players = {}
        self.current_phase = 'idle'
        self.inputs_enabled = False
        self.final_results = None
        self.next_allowed = False   # аудио ответа дослушано: VIP может листать
        self.last_reveal = None     # показанный ответ — для тех, кто вернулся во время показа

game = GameState()
# Обработчики Socket.IO в режиме threading выполняются параллельно, а состояние
# игры общее. Без замка два «Далее» подряд пропускали вопрос, а вход игрока во
# время показа ответа обрывал цикл по игрокам.
state_lock = threading.RLock()
admin_sids = set()


def admin_only(handler):
    """Событие ведущего: при заданном ADMIN_PASSWORD — только после входа."""
    @wraps(handler)
    def wrapper(*args, **kwargs):
        if ADMIN_PASSWORD and request.sid not in admin_sids:
            emit('admin_auth_required', {'retry': False}, to=request.sid)
            return None
        return handler(*args, **kwargs)
    return wrapper

def load_game_questions(gid):
    path = os.path.join(MEDIA_ROOT, f"{gid}-questions.json")
    try:
        if not os.path.exists(path): return []
        with open(path, 'r', encoding='utf-8') as f: return json.load(f)
    except: return []

# --- РОУТЫ ---

@app.route('/')
def index(): return render_template('index.html')

@app.route('/admin')
def admin(): return render_template('admin.html')

# --- SOCKETS: ВХОД И УПРАВЛЕНИЕ ---

@socketio.on('join_game')
def on_join(data):
    data = data or {}
    name = str(data.get('name', '')).strip()[:MAX_PLAYER_NAME_LENGTH]
    player_id = str(data.get('player_id') or '')[:64]

    if not name: return
    if not game.game_id:
        emit('join_error', {'msg': 'Игра ещё не создана!'}, to=request.sid)
        return

    with state_lock:
        # Имя — ключ игрока, постоянный id браузера — доказательство, что это тот
        # же человек. Имя с чужим id занято, даже если его владелец сейчас не в
        # сети (телефон заблокирован): иначе тёзка забирал его очки, а вернувшегося
        # игрока уже не пускало. Освободить имя может ведущий, удалив игрока.
        target_name = name.lower()
        old_sid = None
        for sid, p in game.players.items():
            if p['name'].lower().strip() != target_name:
                continue
            if p.get('player_id') and p.get('player_id') != player_id:
                emit('join_error', {'msg': 'Это имя уже занято — выберите другое.'},
                     to=request.sid)
                return
            old_sid = sid
            break

        prev = game.players.get(old_sid) if old_sid else None
        prev_answer = prev.get('last_answer') if prev else None
        entry = {
            'name': name,
            'score': prev['score'] if prev else 0,
            'last_answer': prev_answer,  # Сохраняем предыдущий ответ при переподключении
            'player_id': player_id or (prev.get('player_id') if prev else ''),
            'online': True,
        }
        if old_sid:
            # Вернувшийся игрок остаётся на своём месте в очереди. VIP — первый
            # вошедший, а pop + вставка уносили его в конец: после переподключения
            # (уснул телефон, обновилась вкладка) кнопка «Далее» уходила другому.
            game.players = {(request.sid if sid == old_sid else sid):
                            (entry if sid == old_sid else p)
                            for sid, p in game.players.items()}
        else:
            game.players[request.sid] = entry

    join_room('players')
    # Имя — как его сохранил сервер (обрезано до MAX_PLAYER_NAME_LENGTH): по нему
    # клиент находит себя в таблице и среди победителей.
    emit('join_success', {'game_id': game.game_id, 'name': name}, to=request.sid)

    with state_lock:
        # Отправляем полное состояние игры переподключившемуся игроку
        emit('game_status', _get_client_state(), to=request.sid)

        # Идёт вопрос — данные вопроса; идёт показ ответа — сам ответ: иначе
        # вернувшийся в этот момент видел экран вопроса и оставался без «Далее».
        if game.current_phase == 'question' and game.current_q_index >= 0:
            current_q = game.questions[game.current_q_index]
            emit('new_question', {
                'index': game.current_q_index + 1,
                'total': len(game.questions),
                'question': current_q.get('question', ''),
                'type': current_q.get('type', 'text'),
                'track_meta': current_q.get('track_meta', ''),
            }, to=request.sid)
        elif game.current_phase == 'answer' and game.last_reveal:
            emit('show_answer_client', {
                **game.last_reveal,
                'leaderboard': _leaderboard(),
                'vip_id': pick_vip_sid(game.players),
                'my_delta': game.last_reveal['deltas'].get(name, 0),
            }, to=request.sid)
            if game.next_allowed:
                # Всем: вернулся VIP — кнопка снова у него, у временного VIP пропадает.
                socketio.emit('allow_next_question', {'vip_id': pick_vip_sid(game.players)},
                              to='players')

        # Если идет фаза ответов и у игрока нет ответа - разрешаем отвечать
        if game.current_phase == 'question' and game.inputs_enabled and not prev_answer:
            emit('allow_answers', to=request.sid)

    _broadcast_admin_info()

@socketio.on('admin_create_game')
@admin_only
def admin_create():
    with state_lock:
        if game.game_id:
            emit('admin_error', {'msg': 'Игра уже создана!'}, to=request.sid)
            return
        game.game_id = ''.join(random.choices(string.ascii_uppercase, k=4))
        game.questions = []
        game.current_q_index = -1
        game.is_active = False
        game.players = {}
        game.current_phase = 'idle'
        socketio.emit('game_reset', to='players')
        _broadcast_admin_info()

@socketio.on('admin_start_round')
@admin_only
def admin_start_round():
    qs = load_game_questions(game.game_id)
    if not qs:
        emit('admin_error', {'msg': 'Вопросы не найдены!'}, to=request.sid)
        return
    with state_lock:
        game.questions = qs
        game.is_active = True
        game.current_q_index = -1
        game.current_phase = 'idle'
        _broadcast_admin_info()

@socketio.on('admin_hard_reset')
@admin_only
def admin_hard_reset():
    with state_lock:
        _hard_reset()


def _hard_reset():
    game.game_id = None
    game.questions = []
    game.players = {}
    game.is_active = False
    game.current_phase = 'idle'
    socketio.emit('game_reset', to='players')
    _broadcast_admin_info()

# --- ГЕНЕРАТОР ---

@socketio.on('gen_start')
@admin_only
def on_gen_start(data):
    data = data or {}
    if not game.game_id:
        emit('gen_log', {'msg': '❌ Сначала создайте игру!'}, to=request.sid)
        emit('gen_finished', {'ok': False, 'msg': 'Сначала создайте игру!'}, to=request.sid)
        return
    try:
        payload = {'game_id': game.game_id, 'token': data.get('token'),
                   'urls': str(data.get('urls', '')).split('\n')}
        resp = requests.post(f"{GENERATOR_URL}/start", json=payload, timeout=2)
        if resp.status_code == 200:
            emit('gen_log', {'msg': '🚀 Задача отправлена...'}, to='admin_room')
            socketio.start_background_task(poll_generator_task)
        else:
            emit('gen_log', {'msg': f'⚠️ Ошибка генератора: {resp.text}'}, to=request.sid)
            emit('gen_finished', {'ok': False, 'msg': resp.text}, to=request.sid)
    except Exception as e:
        emit('gen_log', {'msg': f'❌ Ошибка связи: {e}'}, to=request.sid)
        emit('gen_finished', {'ok': False, 'msg': str(e)}, to=request.sid)


def poll_generator_task():
    """Пересылает лог и прогресс генерации всем окнам ведущего до конца задачи.

    Конец — не только 'finished': 'error', а также 'idle' после того, как задача
    уже шла (генератор перезапустился и потерял её), или если задача так и не
    началась. Долгая недоступность генератора — тоже конец с ошибкой. Раньше
    опрос в этих случаях крутился вечно или молча выходил, и кнопка генерации
    оставалась заблокированной.
    """
    last_log_idx = 0
    fail_count = 0
    idle_polls = 0
    seen_running = False
    while True:
        socketio.sleep(1.5)
        try:
            r = requests.get(f"{GENERATOR_URL}/status", timeout=2).json()
        except Exception as e:
            fail_count += 1
            if fail_count >= 10:
                socketio.emit('gen_finished', {'ok': False,
                                               'msg': f'Генератор не отвечает: {e}'},
                              to='admin_room')
                break
            continue
        fail_count = 0
        logs = r.get('logs') or []
        if len(logs) < last_log_idx:      # генератор перезапустился, лог начат заново
            last_log_idx = 0
        for i in range(last_log_idx, len(logs)):
            socketio.emit('gen_log', {'msg': logs[i]}, to='admin_room')
        last_log_idx = len(logs)
        socketio.emit('gen_progress', {'percent': r.get('progress', 0)}, to='admin_room')

        status = r.get('status')
        if status == 'running' or r.get('is_busy'):
            seen_running = True
            continue
        if status == 'finished':
            with state_lock:
                game.questions = load_game_questions(game.game_id)
            socketio.emit('gen_finished', {'ok': True}, to='admin_room')
            _broadcast_admin_info()
            break
        if status == 'error':
            socketio.emit('gen_finished', {'ok': False, 'msg': 'Генерация завершилась с ошибкой'},
                          to='admin_room')
            break
        idle_polls += 1
        if seen_running or idle_polls >= GENERATOR_IDLE_POLLS_LIMIT:
            socketio.emit('gen_finished', {'ok': False,
                                           'msg': 'Генератор перезапустился, задача потеряна'},
                          to='admin_room')
            break

# --- ГЕЙМПЛЕЙ ---

@socketio.on('admin_next_question')
@admin_only
def admin_next(data=None):
    from_index = (data or {}).get('from_index') if isinstance(data, dict) else None
    _advance_question(from_index)


def _advance_question(from_index=None):
    """Следующий вопрос. ``from_index`` — с какого вопроса нажали «Далее»:
    повторное нажатие (ведущий и VIP одновременно) приходит со старым номером и
    игнорируется, иначе вопрос пропускался."""
    with state_lock:
        if not game.is_active: return
        if from_index is not None and from_index != game.current_q_index:
            # Окно ведущего видит устаревший номер — шлём свежий снимок, иначе и
            # следующее «Далее» ушло бы со старым номером и тоже потерялось.
            _broadcast_admin_info()
            return
        _advance_question_locked()


def _advance_question_locked():
    game.current_q_index += 1
    game.next_allowed = False
    game.last_reveal = None
    if game.current_q_index >= len(game.questions):
        _end_game()
    else:
        game.current_phase = 'question'
        game.inputs_enabled = False
        for pid in game.players: game.players[pid]['last_answer'] = None
        q = game.questions[game.current_q_index]

        audio_url = f"{game.game_id}-media/{q['id']}-1.mp3"

        socketio.emit('new_question', {
            'type': 'text',
            'question': q['question'],
            'index': game.current_q_index + 1,
            'track_meta': q.get('track_meta', ''),
        }, to='players')

        # ВАЖНО: Отправляем в admin_room, чтобы играло у админа, даже если нажал игрок
        socketio.emit('play_audio', {'file': audio_url}, to='admin_room')

        _broadcast_admin_info()

@socketio.on('admin_repeat_question')
@admin_only
def admin_repeat():
    if game.current_q_index >= 0 and game.current_q_index < len(game.questions):
        q = game.questions[game.current_q_index]
        audio_url = f"{game.game_id}-media/{q['id']}-1.mp3"
        emit('play_audio', {'file': audio_url}, to=request.sid)

@socketio.on('admin_audio_finished')
@admin_only
def admin_audio_finished():
    with state_lock:
        if not game.is_active: return

        # 1. Если закончился ВОПРОС -> Разрешаем отвечать
        if game.current_phase == 'question':
            game.inputs_enabled = True
            socketio.emit('allow_answers', to='players')

        # 2. Если закончился ОТВЕТ -> Разрешаем VIP игроку нажать "Далее".
        # VIP — текущий: прежний мог уйти после показа ответа.
        elif game.current_phase == 'answer':
            game.next_allowed = True
            socketio.emit('allow_next_question', {'vip_id': pick_vip_sid(game.players)},
                          to='players')

@socketio.on('submit_answer')
def on_answer(data):
    data = data or {}
    with state_lock:
        if not (game.current_phase == 'question' and game.inputs_enabled):
            return
        if request.sid not in game.players:
            emit('join_error', {'msg': 'Вы не в игре — войдите снова.'}, to=request.sid)
            return
        answer_text = str(data.get('answer') or '').strip()
        is_valid, message = validate_player_answer_text(answer_text)
        if not is_valid:
            emit('answer_validation_error', {
                'msg': message,
                'min_length': MIN_PLAYER_ANSWER_LENGTH,
            }, to=request.sid)
            return
        game.players[request.sid]['last_answer'] = answer_text
        _broadcast_admin_info()
        _maybe_auto_show_locked()


def _maybe_auto_show_locked():
    """Все, кто на связи, ответили — через 3 с показываем ответ.

    Ушедший игрок не ждётся. Проверяется и после ответа, и после ухода игрока:
    иначе последний, кто не ответил и отключился, держал раунд до кнопки ведущего.
    """
    if not (game.current_phase == 'question' and game.inputs_enabled):
        return
    waiting = online_players(game.players)
    if waiting and all(p['last_answer'] for p in waiting.values()):
        game.inputs_enabled = False
        socketio.emit('start_timer', {'seconds': 3}, to='players')
        socketio.start_background_task(_auto_show, game.current_q_index)


def _after_player_left_locked(was_vip):
    """Игрок ушёл или удалён: раунд не должен ждать ни его ответа, ни его кнопки."""
    _maybe_auto_show_locked()
    if was_vip and game.current_phase == 'answer' and game.next_allowed:
        socketio.emit('allow_next_question', {'vip_id': pick_vip_sid(game.players)},
                      to='players')

def _auto_show(q_index):
    socketio.sleep(3)
    with state_lock:
        # Номер вопроса обязателен: если за эти 3 с ведущий уже перешёл дальше,
        # раскрывать нельзя — это был бы следующий вопрос, на который ещё не отвечали.
        if game.current_phase == 'question' and game.current_q_index == q_index:
            with app.app_context(): _reveal()

@socketio.on('admin_show_answer')
@admin_only
def admin_show():
    with state_lock:
        _reveal()

@socketio.on('player_next_question')
def player_next_question():
    with state_lock:
        # VIP — первый из игроков на связи; «Далее» — только после показа ответа.
        if request.sid != pick_vip_sid(game.players): return
        if game.current_phase != 'answer': return
        if not game.next_allowed: return
        if not game.is_active: return
        _advance_question_locked()

def _leaderboard():
    return sorted([{'name': p['name'], 'score': p['score']} for p in list(game.players.values())],
                  key=lambda x: x['score'], reverse=True)


def _reveal():
    if game.current_phase != 'question': return
    game.current_phase = 'answer'
    game.next_allowed = False

    q = game.questions[game.current_q_index]
    correct_clean = normalize_answer_strict(q['answer'])
    deltas = {}

    # 1. Подсчет очков
    for p in list(game.players.values()):
        ans = p.get('last_answer')
        user_clean = normalize_answer_strict(ans)
        is_correct = (user_clean == correct_clean and len(correct_clean) > 0)
        pts = 2 if is_correct else 0
        p['score'] += pts
        deltas[p['name']] = pts

    lb = _leaderboard()

    # 2. Определение VIP и последнего раунда
    vip_sid = pick_vip_sid(game.players)
    is_last_round = (game.current_q_index >= len(game.questions) - 1)
    game.last_reveal = {
        'answer': q['answer'],
        'track_meta': q.get('track_meta', ''),
        'deltas': deltas,
        'is_last': is_last_round,
    }

    # 3. Отправка результатов каждому игроку персонально (с my_delta)
    for sid, p in list(game.players.items()):
        socketio.emit('show_answer_client', {
            **game.last_reveal,
            'leaderboard': lb,
            'vip_id': vip_sid,
            'my_delta': deltas.get(p['name'], 0)  # Персональный результат игрока
        }, to=sid)

    # 4. Аудио ответа
    audio_url = f"{game.game_id}-media/{q['id']}-2.mp3"
    socketio.emit('play_audio', {'file': audio_url}, to='admin_room')

    # 5. ОТПРАВКА ОТВЕТОВ В АДМИНКУ (ЭТО БЫЛО УТЕРЯНО)
    res_list = []
    for p in list(game.players.values()):
        u_cl = normalize_answer_strict(p['last_answer'])
        res_list.append({
            'name': p['name'],
            'answer': p['last_answer'],
            'is_correct': (u_cl == correct_clean and len(correct_clean) > 0)
        })
    socketio.emit('round_results', res_list, to='admin_room')

    _broadcast_admin_info()

@socketio.on('admin_end_game')
@admin_only
def admin_end():
    with state_lock:
        _end_game()


def _end_game():
    game.is_active = False
    game.current_phase = 'finished'
    lb = sorted([{'name': p['name'], 'score': p['score']} for p in list(game.players.values())], key=lambda x:x['score'], reverse=True)
    wins = [p for p in lb if p['score'] == lb[0]['score']] if lb else []
    game.final_results = {'leaderboard': lb, 'winners': wins}
    socketio.emit('game_over', game.final_results)
    _broadcast_admin_info()

# --- УПРАВЛЕНИЕ ОЧКАМИ ---

@socketio.on('admin_give_point')
@admin_only
def admin_give(data):
    pid = (data or {}).get('id')
    with state_lock:
        if pid in game.players:
            game.players[pid]['score'] += 1
            _broadcast_admin_info()

@socketio.on('admin_take_point')
@admin_only
def admin_take(data):
    pid = (data or {}).get('id')
    with state_lock:
        if pid in game.players:
            game.players[pid]['score'] -= 1
            _broadcast_admin_info()

@socketio.on('admin_remove_player')
@admin_only
def admin_remove_player(data):
    """Удаление игрока администратором"""
    pid = (data or {}).get('id')
    with state_lock:
        if pid not in game.players:
            return
        was_vip = pid == pick_vip_sid(game.players)
        del game.players[pid]
        # Отправляем игроку сообщение о сбросе (выкинет его из игры)
        socketio.emit('game_reset', to=pid)
        _broadcast_admin_info()
        _after_player_left_locked(was_vip)

# --- СОХРАНЕННЫЕ ИГРЫ ---

def scan_saved_games():
    games_found = []
    json_files = glob.glob(os.path.join(MEDIA_ROOT, "*-questions.json"))

    for jf in json_files:
        filename = os.path.basename(jf)
        gid = filename.replace("-questions.json", "")
        if not is_valid_game_id(gid):
            continue   # чужой файл в media: id попал бы в onclick админки
        try:
            with open(jf, 'r', encoding='utf-8') as f: qs = json.load(f)
        except: qs = []

        media_path = os.path.join(MEDIA_ROOT, f"{gid}-media")
        media_exists = os.path.exists(media_path)
        valid_files = 0

        if media_exists and qs:
            for q in qs:
                if os.path.exists(os.path.join(media_path, f"{q['id']}-1.mp3")):
                    valid_files += 1

        total = len(qs)
        is_valid = (total > 0) and (valid_files == total) # Валидация для кнопки

        games_found.append({
            'game_id': gid,
            'questions_count': total,
            'valid_count': valid_files,
            'is_valid': is_valid,
            'timestamp': os.path.getmtime(jf)
        })

    games_found.sort(key=lambda x: x['timestamp'], reverse=True)
    return games_found

@socketio.on('admin_get_saved_games')
@admin_only
def on_get_saved_games():
    games_list = scan_saved_games()
    emit('saved_games_list', games_list, to=request.sid)

@socketio.on('admin_load_game')
@admin_only
def on_load_game(data):
    gid = (data or {}).get('game_id')
    if not is_valid_game_id(gid):
        emit('admin_error', {'msg': 'Некорректный id игры'}, to=request.sid)
        return
    qs = load_game_questions(gid)
    if not qs:
        emit('admin_error', {'msg': 'Не удалось загрузить вопросы!'}, to=request.sid)
        return

    with state_lock:
        game.game_id = gid
        game.questions = qs
        game.current_q_index = -1
        game.is_active = False
        game.current_phase = 'idle'
        game.players = {}

    socketio.emit('game_reset', to='players')
    emit('admin_success', {'msg': f'Игра {gid} загружена!'}, to=request.sid)
    _broadcast_admin_info()

@socketio.on('admin_delete_game')
@admin_only
def on_delete_game(data):
    gid = (data or {}).get('game_id')
    if not is_valid_game_id(gid):
        emit('admin_error', {'msg': 'Некорректный id игры'}, to=request.sid)
        return

    if game.game_id == gid:
        with state_lock:
            _hard_reset()

    try:
        json_path = os.path.join(MEDIA_ROOT, f"{gid}-questions.json")
        media_path = os.path.join(MEDIA_ROOT, f"{gid}-media")
        if os.path.exists(json_path): os.remove(json_path)
        if os.path.exists(media_path): shutil.rmtree(media_path)

        emit('admin_success', {'msg': 'Удалено'}, to=request.sid)
        on_get_saved_games()
    except Exception as e:
        emit('admin_error', {'msg': str(e)}, to=request.sid)

# --- ИНФО ---

def _get_client_state():
    if game.current_phase == 'finished': return {'state': 'finished', 'final_results': game.final_results}
    if not game.is_active: return {'state': 'idle'}
    q = game.questions[game.current_q_index] if 0 <= game.current_q_index < len(game.questions) else None
    return {
        'state': game.current_phase,
        'inputs_enabled': game.inputs_enabled,
        'question_data': {
            'question': q['question'],
            'index': game.current_q_index + 1,
            'track_meta': q.get('track_meta', ''),
        } if q else None
    }

def _broadcast_admin_info():
    # Снимок собирается и уходит под замком: иначе снимок, собранный до «Далее»,
    # мог прийти после снимка с новым вопросом, и ведущий видел старый номер.
    with state_lock:
        players_list = []
        for sid, p in list(game.players.items()):
            p_data = p.copy()
            p_data['id'] = sid
            players_list.append(p_data)
        players_list.sort(key=lambda x: x['score'], reverse=True)

        qs = [{'id': q['id'], 'status': ('done' if i<game.current_q_index else ('current' if i==game.current_q_index else 'waiting')), 'type': 'text'} for i,q in enumerate(game.questions)]

        socketio.emit('admin_update', {
            'game_id': game.game_id,
            'players': players_list,
            'questions': qs,
            'game_active': game.is_active,
            'phase': game.current_phase
        }, to='admin_room')

def _admin_key_ok(key):
    if not ADMIN_PASSWORD:
        return True
    return hmac.compare_digest(str(key or '').encode('utf-8'), ADMIN_PASSWORD.encode('utf-8'))


@socketio.on('connect')
def on_connect(auth=None):
    """Окно ведущего присылает пароль при подключении (auth).

    Так вход готов раньше первого события: клики, накопленные клиентом за
    время разрыва, уходят сразу после переподключения и раньше join_admin —
    раньше они отклонялись, а ведущий видел лишний запрос пароля.
    """
    if isinstance(auth, dict) and auth.get('admin') and _admin_key_ok(auth.get('key')):
        admin_sids.add(request.sid)
        join_room('admin_room')


@socketio.on('join_admin')
def join_admin(data=None):
    key = str((data or {}).get('key') or '') if isinstance(data, dict) else ''
    if not _admin_key_ok(key):
        emit('admin_auth_required', {'retry': bool(key)}, to=request.sid)
        return
    admin_sids.add(request.sid)
    join_room('admin_room')
    _broadcast_admin_info()

@socketio.on('disconnect')
def on_disconnect():
    """Отключение: игрок помечается «не в сети», но не удаляется.

    Очки и ответ сохраняются до переподключения; ждать ответа от ушедшего и
    держать за ним кнопку «Далее» игра больше не будет.
    """
    admin_sids.discard(request.sid)
    with state_lock:
        if request.sid in game.players:
            was_vip = request.sid == pick_vip_sid(game.players)
            game.players[request.sid]['online'] = False
            _broadcast_admin_info()
            _after_player_left_locked(was_vip)

if __name__ == '__main__':
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )
