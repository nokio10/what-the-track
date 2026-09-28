FROM python:3.11-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.8.0+cpu
ARG TORCHAUDIO_VERSION=2.8.0+cpu
ARG TORCHVISION_VERSION=0.23.0+cpu

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# ШАГ 1: Мелкие зависимости (чтобы не было тайм-аутов)
RUN pip install --no-cache-dir typing_extensions filelock sympy networkx jinja2 fsspec

# ШАГ 2: Устанавливаем PyTorch + Audio + Vision.
# По умолчанию — CPU wheels, для Nvidia build args можно переопределить.
RUN pip install --no-cache-dir --default-timeout=1000 \
    torch==${TORCH_VERSION} \
    torchaudio==${TORCHAUDIO_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    --index-url ${TORCH_INDEX_URL}

# ШАГ 3: Создаем constraints.txt
# Запрещаем pip'у менять эти версии на что-либо другое
RUN printf "torch==%s\ntorchaudio==%s\ntorchvision==%s\n" \
    "${TORCH_VERSION}" "${TORCHAUDIO_VERSION}" "${TORCHVISION_VERSION}" > constraints.txt

# ШАГ 4: Устанавливаем остальное
COPY requirements.txt .

# Флаг -c constraints.txt гарантирует, что whisperx или audio-separator
# не смогут скачать "обычный" torch/torchvision и сломать сборку.
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url ${TORCH_INDEX_URL} \
    -c constraints.txt

COPY . .
RUN mkdir -p media temp_downloads logs
EXPOSE 5001

CMD ["python", "generator_service.py"]
