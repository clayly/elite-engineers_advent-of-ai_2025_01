# syntax=docker/dockerfile:1
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# (опционально) git может понадобиться некоторым зависимостям
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Обновим инструменты сборки и установим зависимости, нужные для запуска pytest и генерации тестов
RUN pip install --upgrade pip setuptools wheel \
    && pip install "pytest>=8.2.0"

WORKDIR /workspace

# По умолчанию контейнер просто готов для запуска pytest
CMD ["pytest", "-q"]
