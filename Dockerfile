# Базовый образ, можно заменить на pytorch/pytorch:<версия>-cuda...
FROM python:3.10-slim

# Обновим пакеты в базовом образе
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создаем директорию для приложения
WORKDIR /app

# Скопируем файлы зависимостей
COPY requirements.txt /app/requirements.txt

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Скопируем исходный код
COPY app /app/app

# По умолчанию порт 8000
EXPOSE 8000

# Команда запуска: uvicorn на 0.0.0.0:8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
