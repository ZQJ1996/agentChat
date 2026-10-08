from pathlib import Path
from functools import lru_cache

import yaml

PROMPTS_FILE = Path(__file__).with_name("templates.yaml")


@lru_cache
def load_prompts() -> dict[str, str]:
    raw = yaml.safe_load(PROMPTS_FILE.read_text(encoding="utf-8"))
    data: dict[str, object] = raw if isinstance(raw, dict) else {}
    return {
        str(k): str(v).strip()
        for k, v in data.items()
        if isinstance(k, str) and k.startswith("system_")
    }


def get_prompt(name: str) -> str:
    prompts = load_prompts()
    if name not in prompts:
        raise KeyError(f"Unknown prompt: {name}")
    return prompts[name]
