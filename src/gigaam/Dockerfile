# GigaAM v3 (CTC и e2e RNN-T) — соседний сервис генератора.
#
# Отдельный образ, потому что transcribe_longform() тянет pyannote.audio 4.x и
# torchcodec, которые конфликтуют с пинами WhisperX. Связка версий ниже взята
# из asr_lab/docker/Dockerfile.gigaam, где она подобрана перебором; менять её
# по одной версии нельзя — ломается при первом импорте, а не при установке.
FROM python:3.11-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.8.0+cpu
ARG TORCHAUDIO_VERSION=2.8.0+cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/hf \
    GIGAAM_CACHE_DIR=/cache/gigaam
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 \
        torch==${TORCH_VERSION} torchaudio==${TORCHAUDIO_VERSION} \
        --index-url ${TORCH_INDEX_URL}

RUN printf "torch==%s\ntorchaudio==%s\n" "${TORCH_VERSION}" "${TORCHAUDIO_VERSION}" \
        > /app/constraints.txt

# Версии — ровно те, что проверены в работе (сборка 2026-09-26). Незакреплённый
# HEAD GigaAM или новый pyannote.audio ломали бы сборку всего стека.
RUN pip install --no-cache-dir \
        "gigaam[torch] @ git+https://github.com/salute-developers/GigaAM.git@7447938d791c4f3e643386ee22c33777004293a5" \
        pyannote.audio==4.0.7 soundfile==0.14.0 flask==3.1.3 \
        --extra-index-url ${TORCH_INDEX_URL} -c /app/constraints.txt

# pyannote.audio 4.x читает аудио только через torchcodec. Сборка с PyPI ищет
# CUDA 13, cu128 тянет NPP, 0.11.x требует torch 2.9+. CPU-сборка 0.7.0 —
# единственная, что работает с torch 2.8: декодируется только аудио.
RUN pip install --no-cache-dir --force-reinstall --no-deps \
        --index-url https://download.pytorch.org/whl/cpu "torchcodec==0.7.0"

# 3.x не принимает локальный путь к весам, а GigaAM передаёт именно путь.
RUN python -c "import pyannote.audio, sys; \
    sys.exit(0 if int(pyannote.audio.__version__.split('.')[0]) >= 4 else 1)"

COPY app.py .
EXPOSE 5003
CMD ["python", "app.py"]
