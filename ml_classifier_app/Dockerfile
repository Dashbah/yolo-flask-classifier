# Используем многоэтапную сборку
FROM python:3.8-slim as builder

WORKDIR .

# Устанавливаем только необходимые системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Второй этап - финальный образ
FROM python:3.8-slim

WORKDIR .

# Копируем только необходимые файлы из builder
COPY --from=builder /root/.local /root/.local
COPY . .

# Модель монтируем через volume, не включаем в образ
RUN mkdir -p /models

# Очищаем кеш pip
RUN rm -rf /root/.cache/pip

# Добавляем .local в PATH
ENV PATH=/root/.local/bin:$PATH \
    PYTHONPATH=.

ENTRYPOINT ["python", "run.py"]