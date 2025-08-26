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
python -m pip install "langchain>=0.2.16" "langchain-openai>=0.2.0" "langgraph>=0.2.36" "pytest>=8.2.0"
# Или установить зависимости проекта из pyproject:
python -m pip install .
```

## Пример запуска агента

1) Установите переменную окружения с ключом (PowerShell):

```
$env:OPENROUTER_API_KEY = "sk-or-v1-..."
# (необязательно) реферер и название приложения для OpenRouter
$env:OPENROUTER_SITE_URL = "https://your-site.example"
$env:OPENROUTER_APP_NAME = "AI Test Agent"
# (запасной вариант) если нужен fallback
# $env:OPENAI_API_KEY = "sk-..."
```

2) Запустите агента, указав путь к файлу, для которого нужно сгенерировать тесты:

```
python agent.py --file path\to\your_module.py
```

Агент:
- сгенерирует тесты и сохранит их в `tests/test_your_module.py` (или в путь, который вы укажете флагом `--out`);
- соберёт Docker-образ (на основе локального Dockerfile);
- запустит `pytest` внутри контейнера, смонтировав текущую директорию в `/workspace`;
- выведет лог прогона и код возврата.

## Chat-агент (LangGraph + streaming)

В репозитории добавлен асинхронный чат-агент с потоковой передачей токенов на базе LangGraph и LangChain (через OpenRouter).

Установка зависимостей:
- Если вы используете uv: `uv sync` (pyproject уже включает зависимость `langgraph`).
- Или через pip: `python -m pip install .`

Примеры запуска:

1) Интерактивный режим (REPL) с потоковым выводом ответа:
```
python agent_chat.py --model openai/gpt-4o-mini --temperature 0.3 --max-tokens 256 --system "You are a helpful assistant."
```
Далее вводите реплики в консоль. Принудительное завершение: Ctrl+C.

2) Один запрос (single-turn) и выход:
```
python agent_chat.py --model openai/gpt-4o-mini --temperature 0.2 --max-tokens 200 --message "Привет! Что ты умеешь?"
```

Ключевые параметры:
- `--model` — идентификатор модели в OpenRouter (по умолчанию `openai/gpt-4o-mini`).
- `--temperature` — температура выборки модели.
- `--max-tokens` — максимальное число токенов в ответе (ограничения модели применяются).
- `--system` — системный промпт (инструкции для ассистента).
- `--message` — если указан, чат выполнит один запрос и завершится.

OpenRouter:
- Требуется переменная окружения `OPENROUTER_API_KEY` (поддерживается запасной `OPENAI_API_KEY`).
- Базовый URL: `https://openrouter.ai/api/v1`.
- Рекомендуемые заголовки `HTTP-Referer` и `X-Title` настраиваются через `OPENROUTER_SITE_URL` и `OPENROUTER_APP_NAME`.

Streaming:
- Чат использует LangGraph и события `astream_events`, печатая входящие токены по мере генерации.

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
- `/langchain-ai/langgraph` — документация и примеры по LangGraph (построение графов состояний, потоковые события)

## Лицензия
MIT (если требуется, поменяйте под вашу политику).

## Поддержка MCP (MultiServerMCPClient)

Агент поддерживает подключение к нескольким MCP-серверам с помощью библиотеки `langchain-mcp-adapters` (см. Context7 ID: `/langchain-ai/langchain-mcp-adapters`). Это позволяет добавлять внешние инструменты в модель, в т.ч. Git MCP сервер.

Требования:
- Установлен Node.js и `npx` (для Git MCP сервера);
- Python зависимости установлены (пакет `langchain-mcp-adapters` добавлен в зависимости проекта).

Включение MCP (минимально инвазивно — по умолчанию выключено):
- Флаги CLI:
  - `--enable-mcp` — включить поддержку MCP;
  - `--mcp-git` — добавить Git MCP сервер (`@modelcontextprotocol/server-git` через `npx`);
  - `--mcp-extra-servers '<JSON>'` — список дополнительных MCP-серверов (массив объектов с полями `name`, `command`, `args`, `transport`, `env` и т.д.);
  - `--mcp-list-tools` — перед генерацией вывести список доступных MCP-инструментов;
  - `--mcp-bind-tools` — привязать MCP-инструменты к модели (`bind_tools`).

- Эквивалентные переменные окружения:
  - `ENABLE_MCP=1`
  - `MCP_GIT=1`
  - `MCP_EXTRA_SERVERS='[...]'`
  - `MCP_LIST_TOOLS=1`
  - `MCP_BIND_TOOLS=1`

Git MCP сервер:
- Источник: https://github.com/modelcontextprotocol/servers/tree/main/src/git
- Запуск происходит автоматически через `npx @modelcontextprotocol/server-git` с транспортом `stdio`.
- Необязательная переменная окружения `MCP_GIT_REPO` может указать путь к репозиторию, с которым работать (по умолчанию сервер может попытаться использовать текущий контекст, зависит от реализации сервера).

Примеры:

1) Список инструментов без привязки к модели:
```
python agent.py -f path/to/your.py --enable-mcp --mcp-git --mcp-list-tools --skip-run
```

2) Привязать инструменты к модели при генерации:
```
export MCP_GIT_REPO=/path/to/repo
python agent.py -f path/to/your.py --enable-mcp --mcp-git --mcp-bind-tools
```

3) Добавить дополнительные сервера (JSON):
```
python agent.py -f path/to/your.py --enable-mcp \
  --mcp-extra-servers '[{"name":"math","command":"python","args":["math_server.py"],"transport":"stdio"}]'
```

Замечания:
- MCP интеграция опциональна: если флаги и переменные окружения не заданы, поведение агента остаётся прежним.
- При отсутствии пакета `langchain-mcp-adapters` агент продолжит работу без MCP (выведет предупреждение при попытке включить MCP).
- Для Git MCP требуется установленный Node.js и доступный `npx`.

## max tokens and temperature

При слишком малом кол-во токенов ллм может отказаться отвечать.
При малом кол-ве токенов ллм будет стараться ужимать ответ.
При большом или слишком большом кол-ве токенов ллм всё равно будет стараться закруглиться на каком-то пределе, видимо внутреннем.

Более низкая температура даёт статистически более предсказуемый и понятный результат между повторами.
Более высокая температура даёт статистически менее предсказуемый и менее понятный результат между повторами.
При этом в нём также может появиться что-то полезное, что не появилось в ответе с более низкой температурой.

