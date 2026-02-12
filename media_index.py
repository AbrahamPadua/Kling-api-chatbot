import json
import math
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from json_datalayer import HISTORY_PATH

MEDIA_INDEX_KEY = "_media_index"
DEFAULT_PAGE_SIZE = 10


def _load_media_index(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            data = json.loads(content) if content else {}
    except Exception:
        return []
    items = data.get(MEDIA_INDEX_KEY, []) if isinstance(data, dict) else []
    return items if isinstance(items, list) else []


def _save_media_index(path: Path, items: List[Dict[str, str]]) -> None:
    data: Dict[str, object] = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                data = json.loads(content) if content else {}
        except Exception:
            data = {}
    if not isinstance(data, dict):
        data = {}
    data[MEDIA_INDEX_KEY] = items
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        return


def add_media_entry(
    url: str,
    media_type: str,
    thread_id: Optional[str],
    provider: str,
    history_path: Optional[Path] = None,
) -> None:
    path = history_path or Path(HISTORY_PATH)
    items = _load_media_index(path)
    items.insert(
        0,
        {
            "id": str(uuid.uuid4()),
            "url": url,
            "type": media_type,
            "provider": provider,
            "thread_id": thread_id or "",
            "created_at": datetime.utcnow().isoformat() + "Z",
        },
    )
    _save_media_index(path, items)


def render_media_page(
    page: int,
    page_size: int = DEFAULT_PAGE_SIZE,
    history_path: Optional[Path] = None,
) -> str:
    path = history_path or Path(HISTORY_PATH)
    items = _load_media_index(path)
    if not items:
        return "No media saved yet."
    page = max(1, page)
    total_pages = max(1, math.ceil(len(items) / page_size))
    if page > total_pages:
        page = total_pages
    start = (page - 1) * page_size
    end = start + page_size
    page_items = items[start:end]
    lines = [f"Saved media (page {page}/{total_pages}):"]
    for idx, item in enumerate(page_items, start=start + 1):
        media_type = item.get("type", "media")
        url = item.get("url", "")
        provider = item.get("provider", "")
        created = item.get("created_at", "")
        lines.append(f"{idx}. [{media_type}] {url} ({provider} • {created})")
    lines.append("Use /media <page> to navigate pages.")
    return "\n".join(lines)
