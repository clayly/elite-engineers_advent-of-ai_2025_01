# AI агент для автогенерации тестов и запуска их в Docker

Этот агент:
- генерирует pytest-тесты для указанного Python-файла при помощи LangChain (langchain-openai через OpenRouter);
- запускает сгенерированные тесты в локальном Docker-контейнере;
- собирает и выводит результат прогона тестов.

Требования:
- Python 3.13+
- Docker установлен и доступен
- Переменная окружения `OPENROUTER_API_KEY` (для доступа к OpenRouter API; допускается запасной `OPENAI_API_KEY`)

## Установка зависимостей

Вариант A (рекомендуется, если используете uv):

```
uv sync
```

Вариант B (pip):

```
python -m pip install -U pip
# Либо установить зависимости вручную:
python -m pip install "langchain>=0.2.16" "langchain-openai>=0.2.0" "pytest>=8.2.0"
# Или установить зависимости проекта из pyproject:
python -m pip install .
```

## Пример запуска агента

1) Установите переменную окружения с ключом:

```
export OPENROUTER_API_KEY="sk-or-v1-..."
# (необязательно) реферер и название приложения для OpenRouter
export OPENROUTER_SITE_URL="https://your-site.example"
export OPENROUTER_APP_NAME="AI Test Agent"
# (запасной вариант) если нужен fallback
# export OPENAI_API_KEY="sk-..."
```

2) Запустите агента, указав путь к файлу, для которого нужно сгенерировать тесты:

```
python agent.py --file path/to/your_module.py
```

Агент:
- сгенерирует тесты и сохранит их в `tests/test_your_module.py` (или в путь, который вы укажете флагом `--out`);
- соберёт Docker-образ (на основе локального Dockerfile);
- запустит `pytest` внутри контейнера, смонтировав текущую директорию в `/workspace`;
- выведет лог прогона и код возврата.

### Полезные флаги
- `--out` — куда сохранить тесты (по умолчанию: `tests/test_<имя файла>.py`).
- `--model` — модель через OpenRouter (по умолчанию: `openai/gpt-4o-mini`).
- `--temperature` — температура генерации (по умолчанию 0.0).
- `--skip-run` — только сгенерировать тесты, не запускать их.
- `--rebuild-image` — пересобрать Docker-образ без кэша.
- `--timeout` — таймаут (сек) для прогона тестов в контейнере.

## Как это работает
- Агент читает указанный файл, формирует prompt и вызывает ChatOpenAI через LangChain (через OpenRouter API).
- На выходе получает тело pytest-тестов, которое очищается от недопустимых импортов и дополняется фикстурой `target`.
- Для запуска тестов используется Docker: текущая папка монтируется в `/workspace`, запуск происходит командой `pytest`.

## OpenRouter

- Ключ API: `OPENROUTER_API_KEY` (в проекте поддержан запасной `OPENAI_API_KEY`).
- Базовый URL: `https://openrouter.ai/api/v1`.
- Рекомендуемые заголовки: `HTTP-Referer` и `X-Title` — настраиваются через переменные `OPENROUTER_SITE_URL` и `OPENROUTER_APP_NAME`.
- Документация: https://openrouter.ai/docs/community/lang-chain

## Пример

Пусть есть файл `src/utils.py`. Запуск:

```
python agent.py -f src/utils.py
```

Выходные тесты появятся в `tests/test_utils.py`, дальше агент сам запустит их в контейнере.

## Примечание про Context7
В процессе разработки использовалась платформа Context7 для получения документации и примеров кода по библиотекам. Для LangChain основная документация доступна по идентификатору:

- `/langchain-ai/langchain` — базовые концепции и примеры
- `langchain-openai` — интеграция с OpenAI через LangChain

## Лицензия
MIT (если требуется, поменяйте под вашу политику).