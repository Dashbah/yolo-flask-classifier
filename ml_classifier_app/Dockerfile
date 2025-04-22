# Используем многоэтапную сборку
FROM python:3.9-slim as builder

WORKDIR /app

# Устанавливаем только необходимые для сборки пакеты
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Копируем только requirements.txt сначала для лучшего кэширования
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Копируем остальные файлы
COPY . .

# Финальный образ
FROM python:3.9-slim

WORKDIR /app

# Копируем только необходимые файлы из builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app /app

# Убедимся, что скрипты в .local доступны в PATH
ENV PATH=/root/.local/bin:$PATH
ENV FLASK_APP=app.py
ENV PYTHONPATH=/app

# Уменьшаем время запуска Flask
ENV FLASK_ENV=production

EXPOSE 5000

# Используем gunicorn для production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]