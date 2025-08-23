#!/usr/bin/env python3
"""
AI агент: генерирует pytest-тесты для указанного Python-файла с помощью LangChain + OpenRouter
и запускает их в локальном Docker-контейнере, затем выводит результат прогона.

Требования окружения для запуска агента (на хосте):
- Python 3.13+
- Установленные зависимости (см. pyproject.toml)
- Переменная окружения OPENROUTER_API_KEY (или OPENAI_API_KEY в качестве запасного варианта)
- Docker установлен и доступен

Пример запуска см. в README.md
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


HEADER_FIXTURE = """
# Авто-добавленный хедер для корректного импорта целевого модуля по файловому пути.
import importlib.util
import inspect
import sys as _sys
from pathlib import Path as _Path
import os as _os
import pytest

@pytest.fixture(scope="session")

def target():
    # Фикстура загружает модуль под именем 'target' из пути, переданного через переменную окружения TARGET_FILE
    file_path = _os.environ.get("TARGET_FILE")
    if not file_path:
        pytest.fail("Переменная окружения TARGET_FILE не установлена")
    file_path = _Path(file_path).resolve()
    if not file_path.exists():
        pytest.fail(f"Файл не найден: {file_path}")
    module_name = "target"
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        pytest.fail("Не удалось создать спецификацию модуля")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _sys.modules[module_name] = module
    return module
""".lstrip()


SYSTEM_PROMPT = (
    "Ты — помощник-разработчик. Твоя задача — написать тесты на pytest для предоставленного Python-кода. "
    "Требования: \n"
    "- Пиши только код тестов без лишних комментариев и объяснений.\n"
    "- Не импортируй целевой модуль напрямую; используй фикстуру 'target' (она уже будет в файле).\n"
    "- Можно импортировать только стандартные библиотеки и 'pytest'.\n"
    "- Предпочтительно покрыть ключевые функции/классы, граничные случаи, исключения.\n"
    "- Не обращайся к внешним ресурсам, файлам или сети.\n"
    "- Используй читаемые имена тестов.\n"
)

USER_PROMPT_TEMPLATE = (
    "Сгенерируй pytest-тесты для следующего Python-файла. \n"
    "Путь к файлу: {target_path}\n\n"
    "Содержимое файла:\n" + "{code}\n\n"
    "Формат ответа: только содержимое тестового файла (функции test_*, возможные фикстуры), без обрамляющих тройных кавычек."
)


@dataclass
class AgentConfig:
    target_file: Path
    out_tests: Optional[Path]
    model: str
    temperature: float = 0.0
    max_tokens: int = 3000
    docker_image: str = "ai-agent-pytest:latest"
    rebuild_image: bool = False
    skip_run: bool = False
    timeout: int = 900  # seconds for docker run


def read_file(path: Path, max_chars: int = 60_000) -> str:
    text = path.read_text(encoding="utf-8")
    if len(text) > max_chars:
        # Обрезаем очень длинные файлы, чтобы не переполнить контекст
        return text[:max_chars]
    return text


def init_llm(model: str, temperature: float) -> ChatOpenAI:
    # Инициализация LLM через OpenRouter (см. https://openrouter.ai/docs/community/lang-chain)
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = "https://openrouter.ai/api/v1"

    # Рекомендуемые дополнительные заголовки
    default_headers = {}
    site = os.environ.get("OPENROUTER_SITE_URL")
    app = os.environ.get("OPENROUTER_APP_NAME")
    if site:
        default_headers["HTTP-Referer"] = site
    if app:
        default_headers["X-Title"] = app

    kwargs = {
        "model": model,
        "temperature": temperature,
        "base_url": base_url,
    }
    if api_key:
        kwargs["api_key"] = api_key
    if default_headers:
        kwargs["default_headers"] = default_headers

    return ChatOpenAI(**kwargs)


def build_chain(model: str, temperature: float):
    llm = init_llm(model, temperature)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT_TEMPLATE),
    ])
    return prompt | llm


def extract_code_blocks(s: str) -> str:
    """Извлекает python-код из ответа. Если есть ```python блоки — объединяет их. Иначе возвращает как есть."""
    blocks = []
    for m in re.finditer(r"```(?:python)?\n(.*?)```", s, flags=re.DOTALL | re.IGNORECASE):
        blocks.append(m.group(1).strip())
    if blocks:
        return "\n\n".join(blocks).strip()
    return s.strip()


def sanitize_tests(code: str) -> str:
    # Удалим потенциальные прямые импорты целевого модуля, навязанные моделью
    lines = code.splitlines()
    filtered = []
    for ln in lines:
        if re.match(r"\s*from\s+\S+\s+import\s+\S+", ln) or re.match(r"\s*import\s+\S+", ln):
            # Разрешим только pytest
            if re.match(r"\s*import\s+pytest\b", ln) or re.match(r"\s*from\s+pytest\s+import\s+", ln):
                filtered.append(ln)
            # Пропускаем остальные импорты (включая модуль под тестом)
            continue
        filtered.append(ln)
    code = "\n".join(filtered).strip()
    # Убедимся, что pytest импортирован хотя бы один раз
    if "import pytest" not in code and "from pytest import" not in code:
        code = "import pytest\n\n" + code
    return code


def compose_test_module(generated_body: str) -> str:
    body = sanitize_tests(generated_body)
    return HEADER_FIXTURE + "\n\n" + body + "\n"


def default_tests_path(target_file: Path) -> Path:
    tests_dir = Path("tests")
    stem = target_file.stem
    return tests_dir / f"test_{stem}.py"


def generate_tests(config: AgentConfig) -> Path:
    code = read_file(config.target_file)
    chain = build_chain(config.model, config.temperature)
    resp = chain.invoke({
        "target_path": str(config.target_file),
        "code": code,
    })
    raw = getattr(resp, "content", str(resp))
    body = extract_code_blocks(raw)
    test_module = compose_test_module(body)

    out_path = config.out_tests or default_tests_path(config.target_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(test_module, encoding="utf-8")
    return out_path


def docker_build(image: str, rebuild: bool = False) -> None:
    args = ["docker", "build", "-t", image, "."]
    if rebuild:
        # Принудительная пересборка через --no-cache
        args = ["docker", "build", "--no-cache", "-t", image, "."]
    subprocess.run(args, check=True)


def docker_run_pytest(image: str, test_path: Path, target_file: Path, timeout: int) -> Tuple[int, str]:
    # Запуск pytest с передачей пути к целевому файлу
    cwd = Path.cwd()
    try:
        rel_test = test_path.relative_to(cwd)
        test_arg = rel_test.as_posix()
    except ValueError:
        # Если путь вне рабочей директории, используем абсолютный путь внутри /workspace
        test_arg = f"/workspace/{test_path.as_posix()}"

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{str(cwd)}:/workspace",
        "-w", "/workspace",
        "-e", f"TARGET_FILE=/workspace/{target_file.as_posix()}",
        image,
        "pytest", "-q", test_arg, "--maxfail=1", "--disable-warnings"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    return proc.returncode, proc.stdout


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ИИ агент: генерирует тесты и запускает их в Docker")
    parser.add_argument("--file", "-f", required=True, help="Путь к целевому .py файлу")
    parser.add_argument("--out", "-o", help="Куда записать сгенерированные тесты (по умолчанию tests/test_<name>.py)")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="Модель через OpenRouter для генерации тестов (langchain-openai)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--skip-run", action="store_true", help="Только сгенерировать тесты, не запускать их")
    parser.add_argument("--rebuild-image", action="store_true", help="Пересобрать Docker-образ без кэша")
    parser.add_argument("--timeout", type=int, default=900, help="Таймаут (сек) для запуска pytest в контейнере")

    args = parser.parse_args(argv)

    target_file = Path(args.file)
    if not target_file.exists():
        print(f"[ERROR] Файл не найден: {target_file}", file=sys.stderr)
        return 2

    out_tests = Path(args.out) if args.out else None
    config = AgentConfig(
        target_file=target_file,
        out_tests=out_tests,
        model=args.model,
        temperature=args.temperature,
        rebuild_image=args.rebuild_image,
        skip_run=args.skip_run,
        timeout=args.timeout,
    )

    # Проверим ключи для OpenRouter
    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("[WARN] Переменная окружения OPENROUTER_API_KEY не установлена (и отсутствует запасной OPENAI_API_KEY). Генерация тестов может не сработать.", file=sys.stderr)

    print(f"[INFO] Генерация тестов для: {config.target_file}")
    try:
        test_path = generate_tests(config)
    except Exception as e:
        print(f"[ERROR] Ошибка генерации тестов: {e}", file=sys.stderr)
        return 3

    print(f"[INFO] Тесты сохранены: {test_path}")

    if config.skip_run:
        return 0

    # Соберём и запустим Docker
    print(f"[INFO] Сборка Docker-образа: {config.docker_image}")
    try:
        docker_build(config.docker_image, rebuild=config.rebuild_image)
    except FileNotFoundError:
        print("[ERROR] Docker не найден. Установите Docker и повторите.", file=sys.stderr)
        return 4
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Ошибка сборки Docker-образа: {e}", file=sys.stderr)
        return 5

    print(f"[INFO] Запуск тестов в контейнере...")
    try:
        code, output = docker_run_pytest(config.docker_image, test_path, config.target_file, config.timeout)
    except subprocess.TimeoutExpired:
        print("[ERROR] Таймаут прогона тестов в контейнере", file=sys.stderr)
        return 6
    except subprocess.CalledProcessError as e:
        # Мы не используем check=True, поэтому сюда обычно не попадём
        print(f"[ERROR] Ошибка запуска контейнера: {e}", file=sys.stderr)
        return 7

    print("\n===== Pytest Output (container) =====\n")
    print(output)
    print("===== End Output =====\n")

    if code == 0:
        print("[SUCCESS] Все тесты прошли успешно.")
    else:
        print(f"[FAIL] Тесты завершились с кодом {code}.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
