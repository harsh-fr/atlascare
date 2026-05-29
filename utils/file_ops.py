import json
import os
import tempfile
from typing import Any


def atomic_json_write(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dir_name = os.path.dirname(path)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=dir_name, delete=False, suffix=".tmp"
    ) as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def sort_by_recency(items: list[dict], field: str = "created_at") -> list[dict]:
    return sorted(items, key=lambda x: x.get(field, ""), reverse=True)
