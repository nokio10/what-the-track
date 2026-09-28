# -*- coding: utf-8 -*-
"""GigaAM v3 как соседний сервис генератора.

Зачем отдельный контейнер: GigaAM для целого трека нужен ``transcribe_longform``,
а он тянет pyannote.audio 4.x и torchcodec, которые конфликтуют с пинами
WhisperX. В asr_lab они по той же причине живут в разных образах.

API:
    GET  /health                      -> {"ok": true, "ready": bool, "error": str|null,
                                          "models": [...загруженные сейчас]}
    POST /transcribe {"path", "model"} -> {"words": [{"word", "start", "end"}]}
                                          или {"error": "..."} с кодом 4xx/5xx

``path`` — путь к аудио внутри общего тома ``/app/media``; файлы вне него сервис
не читает. ``model`` — ``v3_ctc`` или ``v3_e2e_rnnt``. Запросы выполняются по
одному: обе модели делят одну видеокарту.

Модели живут в дочернем процессе. Его запускает первый запрос, а после
GIGAAM_IDLE_UNLOAD_SEC простоя (по умолчанию минута) он завершается: видеопамять
освобождается целиком, вместе с CUDA-контекстом. Простое удаление моделей в
долгоживущем процессе оставляло бы сотни мегабайт.

Разбор ответа GigaAM повторяет адаптер лабы
(asr_lab/asr_lab/transcribe/backends/gigaam_be.py), на котором сняты цифры.
"""
from __future__ import annotations

import multiprocessing
import os
import threading
import time
import wave
from pathlib import Path

from flask import Flask, jsonify, request

MEDIA_ROOT = Path(os.environ.get("MEDIA_ROOT", "/app/media")).resolve()
MODELS = ("v3_ctc", "v3_e2e_rnnt")



def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


# Через сколько секунд простоя процесс с моделями завершается.
IDLE_UNLOAD_SEC = _env_float("GIGAAM_IDLE_UNLOAD_SEC", 60.0)
# Сколько ждать ответа процесса с моделями на один запрос.
WORKER_TIMEOUT_SEC = _env_float("GIGAAM_WORKER_TIMEOUT_SEC", 900.0)

app = Flask(__name__)
# Запросы к моделям по одному; под этим же замком процесс выгружается.
_lock = threading.Lock()
_worker = {"proc": None, "conn": None, "last_used": 0.0, "models": []}


class TranscribeError(RuntimeError):
    """Отказ распознавания с причиной, пригодной для ответа клиенту."""


def _worker_main(conn):
    """Дочерний процесс с моделями: только он трогает видеокарту."""
    import gigaam
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    while True:
        try:
            request_data = conn.recv()
        except (EOFError, OSError):
            return
        if request_data is None:
            return
        name, path = request_data
        try:
            if name not in models:
                models[name] = gigaam.load_model(
                    name,
                    device=device,
                    fp16_encoder=device == "cuda",
                    download_root=os.environ.get("GIGAAM_CACHE_DIR") or None,
                )
            result = models[name].transcribe_longform(path, word_timestamps=True)
            conn.send(("ok", words_from_longform(result), sorted(models)))
        except Exception as e:
            conn.send(("error", "%s: %s" % (type(e).__name__, e), sorted(models)))


def _stop_worker(reason: str = "") -> None:
    """Завершает процесс с моделями. Вызывать под _lock."""
    proc, conn = _worker["proc"], _worker["conn"]
    _worker.update(proc=None, conn=None, models=[])
    if conn is not None:
        try:
            conn.send(None)
        except (OSError, ValueError):
            pass
    if proc is not None:
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
    if conn is not None:
        conn.close()
    if proc is not None and reason:
        app.logger.warning("GigaAM выгружен из памяти: %s", reason)


def _worker_transcribe(name: str, path: str) -> list:
    """Слова от процесса с моделями; при необходимости запускает его. Под _lock."""
    proc = _worker["proc"]
    if proc is None or not proc.is_alive():
        if proc is not None:
            _stop_worker()
        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(target=_worker_main, args=(child_conn,),
                           name="gigaam-models", daemon=True)
        proc.start()
        child_conn.close()
        _worker.update(proc=proc, conn=parent_conn, models=[])
    conn = _worker["conn"]
    try:
        conn.send((name, path))
        if not conn.poll(WORKER_TIMEOUT_SEC):
            _stop_worker("нет ответа за %d с" % WORKER_TIMEOUT_SEC)
            raise TranscribeError("GigaAM не ответил за %d с" % WORKER_TIMEOUT_SEC)
        status, payload, models = conn.recv()
    except (EOFError, OSError) as e:
        _stop_worker("процесс с моделями завершился")
        raise TranscribeError("процесс с моделями GigaAM завершился "
                              "(не хватило памяти?): %s" % e) from None
    finally:
        _worker["last_used"] = time.time()
    _worker["models"] = models
    if status != "ok":
        raise TranscribeError(payload)
    return payload


# Распознавание: в тестах подменяется, чтобы не запускать модели.
_transcribe_impl = _worker_transcribe


def _describe(error: Exception) -> str:
    if isinstance(error, TranscribeError):
        return str(error)
    return "%s: %s" % (type(error).__name__, error)


def _unload_when_idle() -> None:
    """Раз в 5 с: процесс с моделями простаивает дольше IDLE_UNLOAD_SEC — завершить."""
    while True:
        time.sleep(5)
        with _lock:
            if (_worker["proc"] is not None
                    and time.time() - _worker["last_used"] >= IDLE_UNLOAD_SEC):
                _stop_worker("простой %d с" % IDLE_UNLOAD_SEC)


def _num(value):
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def _get(item, *names):
    for name in names:
        if isinstance(item, dict) and name in item:
            return item[name]
        if not isinstance(item, dict) and hasattr(item, name):
            return getattr(item, name)
    return None


def words_from_longform(result) -> list:
    """Слова с абсолютными таймкодами из ``LongformTranscriptionResult``.

    Документация GigaAM не фиксирует, абсолютные у слов времена или локальные
    для сегмента. Если первое слово начинается заметно раньше своего сегмента,
    времена локальные и сдвигаются всем окном разом.
    """
    segments = getattr(result, "segments", None)
    if segments is None:
        segments = result if isinstance(result, (list, tuple)) else []
    out = []
    for segment in segments:
        seg_start = _num(getattr(segment, "start", None))
        words = []
        for word in getattr(segment, "words", None) or []:
            token = str(_get(word, "text", "word", "w") or "").strip()
            if token:
                words.append({"word": token,
                              "start": _num(_get(word, "start", "s")),
                              "end": _num(_get(word, "end", "e"))})
        first = next((w["start"] for w in words if w["start"] is not None), None)
        if seg_start is not None and first is not None and first < seg_start - 0.5:
            for w in words:
                w["start"] = _num((w["start"] or 0.0) + seg_start)
                w["end"] = _num((w["end"] or 0.0) + seg_start)
        out.extend(words)
    return out


def _safe_path(raw: str) -> Path | None:
    try:
        path = Path(raw).resolve()
    except (OSError, ValueError):
        return None
    if MEDIA_ROOT != path and MEDIA_ROOT not in path.parents:
        return None
    return path if path.is_file() else None


# Итог последнего пробного прогона: текст исключения или None.
_WARMUP = {"error": None}
# Пауза между повторами самопроверки, если она не прошла.
try:
    WARMUP_RETRY_SEC = float(os.environ.get("GIGAAM_WARMUP_RETRY_SEC") or 300)
except ValueError:
    WARMUP_RETRY_SEC = 300.0


def warm_up():
    """Пробный прогон на 3 с тишины при старте.

    Загружает VAD pyannote: если условия gated-модели не приняты или токен
    отозван, это выясняется сразу, а не отказом на каждом треке. Тишина
    проходит примерно за 10 с и даёт пустой результат. Модель после прогона
    выгрузится вместе с процессом через IDLE_UNLOAD_SEC простоя.
    """
    sample = "/tmp/gigaam_warmup.wav"
    with wave.open(sample, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(b"\x00\x00" * 16000 * 3)
    try:
        with _lock:
            _transcribe_impl(MODELS[0], sample)
        _WARMUP["error"] = None
    except Exception as e:
        _WARMUP["error"] = _describe(e)


def _retry_warm_up():
    """Повторяет самопроверку, пока она не пройдёт.

    Временный сбой при старте (нет сети до HuggingFace) не должен выключать
    согласие до перезапуска контейнера.
    """
    while _WARMUP["error"]:
        time.sleep(WARMUP_RETRY_SEC)
        warm_up()


@app.get("/health")
def health():
    # ready — не просто «есть токен»: при старте прогоняется самопроверка, и
    # генератор сразу ведёт треки старым путём, если распознавание не работает.
    error = None
    if not (os.environ.get("HF_TOKEN") or "").strip():
        error = "нет HF_TOKEN"
    elif _WARMUP["error"]:
        error = "самопроверка не прошла: %s" % _WARMUP["error"]
    loaded = list(_worker["models"]) if _worker["proc"] is not None else []
    return jsonify({"ok": True, "ready": error is None, "error": error,
                    "models": loaded, "idle_unload_sec": IDLE_UNLOAD_SEC})


@app.post("/transcribe")
def transcribe():
    payload = request.get_json(silent=True) or {}
    name = payload.get("model")
    if name not in MODELS:
        return jsonify({"error": "model должна быть одной из %s" % ", ".join(MODELS)}), 400
    path = _safe_path(str(payload.get("path") or ""))
    if path is None:
        return jsonify({"error": "файл не найден внутри %s" % MEDIA_ROOT}), 400
    if not (os.environ.get("HF_TOKEN") or "").strip():
        return jsonify({"error": "нужен HF_TOKEN: transcribe_longform режет аудио "
                                 "через pyannote/segmentation-3.0"}), 503

    t0 = time.time()
    try:
        with _lock:
            words = _transcribe_impl(name, str(path))
    except Exception as e:
        # Причина — в JSON: генератор покажет её в логе вместо «500 Server Error».
        app.logger.exception("transcribe_longform failed")
        return jsonify({"error": _describe(e)}), 500
    return jsonify({"words": words, "model": name, "elapsed": round(time.time() - t0, 2)})


if __name__ == "__main__":
    # Самопроверка при старте; модели после неё выгрузятся через минуту простоя.
    if os.environ.get("GIGAAM_PRELOAD", "1") == "1" and (os.environ.get("HF_TOKEN") or "").strip():
        warm_up()
        if _WARMUP["error"]:
            threading.Thread(target=_retry_warm_up, daemon=True).start()
    threading.Thread(target=_unload_when_idle, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5003")), threaded=True)
