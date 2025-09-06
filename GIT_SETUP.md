# Настройка Git репозитория для System Audio Copilot

## 🚀 Быстрая настройка

### Вариант 1: Автоматическая настройка
Запустите batch файл:
```cmd
setup_git.bat
```

### Вариант 2: Ручная настройка

#### 1. Инициализация Git репозитория
```cmd
git init
```

#### 2. Добавление файлов
```cmd
git add .
```

#### 3. Первый коммит
```cmd
git commit -m "Initial commit: System Audio Copilot - Python CLI tool for live system audio transcription with AI assistance"
```

#### 4. Настройка основной ветки
```cmd
git branch -M main
```

## 📋 Следующие шаги

### 1. Создайте репозиторий на GitHub
1. Перейдите на https://github.com
2. Нажмите "New repository"
3. Название: `system_audio_copilot`
4. Описание: `Python CLI tool for live system audio transcription with AI assistance`
5. Выберите "Public"
6. **НЕ** добавляйте README, .gitignore или лицензию (они уже есть)
7. Нажмите "Create repository"

### 2. Свяжите локальный проект с GitHub
```cmd
git remote add origin https://github.com/YOUR_USERNAME/system_audio_copilot.git
```
Замените `YOUR_USERNAME` на ваш GitHub username.

### 3. Загрузите код на GitHub
```cmd
git push -u origin main
```

## ✅ Ожидаемый результат

После выполнения всех шагов:
- ✅ Публичный репозиторий `system_audio_copilot` на GitHub
- ✅ Весь код проекта загружен
- ✅ Файл `.env` НЕ загружен (игнорируется)
- ✅ Папка `.venv/` НЕ загружена (игнорируется)
- ✅ Кэш Python игнорируется

## 📁 Структура репозитория

```
system_audio_copilot/
├── .gitignore              # Игнорирование .env, .venv, кэша
├── README.md               # Подробная документация
├── QUICKSTART.md           # Быстрый старт
├── GIT_SETUP.md           # Эта инструкция
├── requirements.txt        # Python зависимости
├── env_example.txt         # Пример конфигурации
├── run.bat                # Скрипт запуска для Windows
├── setup_git.bat          # Скрипт настройки Git
├── main.py                # Основной CLI интерфейс
├── audio_capture.py       # Захват системного звука
├── stt.py                 # Транскрибация через Whisper
└── llm.py                 # AI подсказки через GPT
```

## 🔧 Полезные Git команды

```cmd
# Проверить статус
git status

# Посмотреть историю коммитов
git log --oneline

# Добавить изменения
git add .
git commit -m "Описание изменений"

# Отправить изменения на GitHub
git push

# Получить изменения с GitHub
git pull
```

## ⚠️ Важные моменты

- **Никогда не коммитьте** файл `.env` с API ключами
- **Всегда проверяйте** `.gitignore` перед коммитом
- **Используйте описательные** сообщения коммитов
- **Делайте коммиты** для каждого значимого изменения
