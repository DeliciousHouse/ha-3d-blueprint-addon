import importlib.util
from pathlib import Path
from unittest.mock import patch


def load_engine_module():
    engine_path = Path(__file__).resolve().parents[1] / "blueprint_engine" / "rootfs" / "usr" / "bin" / "engine.py"
    spec = importlib.util.spec_from_file_location("engine", engine_path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    return engine


def test_get_configured_port_defaults_when_missing_options():
    engine = load_engine_module()
    with patch("builtins.open", side_effect=FileNotFoundError()):
        port = engine.get_configured_port()
    assert port == 8124
