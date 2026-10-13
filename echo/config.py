import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_DIR / ".env"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and os.environ.get(key) is None:
            os.environ[key] = value


def _get(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


class Settings:
    def __init__(self) -> None:
        _load_dotenv(ENV_PATH)
        self.gemini_api_key = _get("GEMINI_API_KEY", "").strip()
        self.whisper_model = (_get("WHISPER_MODEL", "base") or "base").strip()
        self.gemini_model = (_get("GEMINI_MODEL", "gemini-3.8-flash") or "gemini-3.8-flash").strip()
        self.whisper_device = "cpu"
        self.whisper_compute = "int8"
        self.sample_rate = 16000

    @property
    def has_api_key(self) -> bool:
        return bool(self.gemini_api_key)