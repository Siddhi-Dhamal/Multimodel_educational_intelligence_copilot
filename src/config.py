import os
from pathlib import Path


def load_dotenv(project_root: str | Path | None = None) -> None:
    """Load environment variables from a local .env file if it exists."""
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parent.parent
    env_path = root / ".env"

    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


def get_env(name: str, default: str | None = None, project_root: str | Path | None = None) -> str | None:
    load_dotenv(project_root)
    return os.environ.get(name, default)
