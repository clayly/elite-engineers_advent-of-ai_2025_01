# Авто-добавленный хедер для корректного импорта целевого модуля по файловому пути.
import importlib.util
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

def test_returns_three(target):
    assert target.get_number_three() == 3

def test_return_type_is_integer(target):
    assert isinstance(target.get_number_three(), int)

def test_does_not_return_string(target):
    assert target.get_number_three() != "3"

# =================================== FAILURES ===================================
# ________________________ test_function_has_no_arguments ________________________
#
# target = <module 'target' from '/workspace/function.py'>
#
#     def test_function_has_no_arguments(target):
# >       sig = inspect.signature(target.get_number_three)
#               ^^^^^^^
# E       NameError: name 'inspect' is not defined. Did you forget to import 'inspect'?
#
# tests/test_function.py:40: NameError
# =========================== short test summary info ============================
# FAILED tests/test_function.py::test_function_has_no_arguments - NameError: na...
# !!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!
# 1 failed, 3 passed in 0.02s
#
# ===== End Output =====
#
# [FAIL] Тесты завершились с кодом 1.

def test_function_has_no_arguments(target):
    sig = inspect.signature(target.get_number_three)
    assert len(sig.parameters) == 0
