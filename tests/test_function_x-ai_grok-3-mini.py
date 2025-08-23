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


import pytest

def test_get_number_three_returns_three(target):
    result = target.get_number_three()
    assert result == 3

def test_get_number_three_does_not_accept_arguments(target):
    try:
        target.get_number_three(1)  # Should raise TypeError
    except TypeError:
        pass  # Expected
    else:
        assert False, "Function should raise TypeError"
