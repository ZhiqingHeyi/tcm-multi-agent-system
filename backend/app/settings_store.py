import json
import threading
from typing import Any

from .config import BACKEND_DIR, settings

_STORE_PATH = BACKEND_DIR / "data" / "llm_settings.json"
_lock = threading.Lock()
_KEYS = ("llm_base_url", "llm_api_key", "llm_model_fast", "llm_model_pro")

def _read_file() -> dict[str, Any]:
    if _STORE_PATH.exists():
        try:
            return json.loads(_STORE_PATH.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}
    return {}

def _write_file(data: dict[str, Any]) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STORE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def load_runtime() -> None:
    data = _read_file()
    for key in _KEYS:
        if data.get(key):
            setattr(settings, key, data[key])

def update_runtime(base_url, api_key, model_fast, model_pro) -> dict[str, Any]:
    with _lock:
        data = _read_file()
        pairs = {
            "llm_base_url": base_url.strip() if base_url is not None else None,
            "llm_api_key": api_key.strip() if api_key else None,
            "llm_model_fast": model_fast.strip() if model_fast else None,
            "llm_model_pro": model_pro.strip() if model_pro else None,
        }
        for key, value in pairs.items():
            if value is not None:
                setattr(settings, key, value)
                data[key] = value
        _write_file(data)
    return public_settings()

def public_settings() -> dict[str, Any]:
    key = settings.llm_api_key
    masked = (key[:3] + "…" + key[-4:]) if len(key) > 8 else ("已设置" if key else "")
    return {
        "llm_base_url": settings.llm_base_url,
        "llm_model_fast": settings.llm_model_fast,
        "llm_model_pro": settings.llm_model_pro,
        "api_key_masked": masked,
        "has_api_key": bool(key),
        "configured": bool(settings.llm_base_url and key and settings.llm_model_fast and settings.llm_model_pro),
    }
