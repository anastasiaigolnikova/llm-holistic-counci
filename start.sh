#!/bin/bash

# LLM Council Startup Script

echo "🏛️  Запуск LLM Council..."
echo ""

# Проверка .env файла
if [ ! -f ".env" ]; then
    echo "⚠️  Файл .env не найден!"
    echo "   Создаю из шаблона .env.example..."
    cp .env.example .env
    echo ""
    echo "📝 ВАЖНО: Отредактируйте .env и добавьте ваши API ключи:"
    echo "   - OpenAI: https://platform.openai.com/api-keys"
    echo "   - Anthropic: https://console.anthropic.com/settings/keys"
    echo "   - Google (БЕСПЛАТНО!): https://aistudio.google.com/app/apikey"
    echo ""
    echo "   Можно начать только с Google — он бесплатный!"
    echo ""
    exit 1
fi

# Проверка Python зависимостей
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "📦 Устанавливаю зависимости..."
    pip3 install -r requirements.txt --user
    echo ""
fi

# Запуск сервера
echo "🚀 Сервер запускается на http://localhost:8000"
echo "   Нажмите Ctrl+C для остановки"
echo ""

python3 server.py

