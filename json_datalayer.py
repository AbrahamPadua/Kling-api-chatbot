"""
Chainlit Data Layer backed by a local JSON file (chat_history.json).

This implements the BaseDataLayer interface so Chainlit can power its built-in
Threads sidebar, without any external database. It stores threads and steps in a
single JSON file. The implementation is intentionally minimal but complete
enough for the sidebar to function.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from chainlit.types import PaginatedResponse, ThreadFilter, PageInfo, Pagination
from chainlit.user import PersistedUser
from chainlit.logger import logger


HISTORY_PATH = Path(__file__).with_name("chat_history.json")


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


class JsonDataLayer:
    def __init__(self, file_path: Optional[str] = None):
        self.file_path = file_path or str(HISTORY_PATH)
        self._last_identifier: Optional[str] = None
        self._last_user_id: Optional[str] = None
        self._migrated_user_ids: bool = False
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _load_data(self) -> Dict[str, Any]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read()
                obj = json.loads(content) if content else {}
        except Exception:
            return {}

        # Legacy format support: prior versions stored a list of message pairs.
        # Convert that list into a single thread with steps so the data layer
        # API always receives a dictionary keyed by thread_id.
        if isinstance(obj, list):
            data = self._migrate_legacy_list(obj)
            self._save_data(data)
            return data

        if not isinstance(obj, dict):
            return {}

        self._prune_empty_threads(obj)
        return obj

    def _save_data(self, data: Dict[str, Any]) -> None:
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _migrate_thread_user_ids(self, target_user_id: str) -> None:
        if not target_user_id:
            return
        data = self._load_data()
        legacy_ids = {"me", "user-1", "", None}
        changed = False
        for thread_id, thread in data.items():
            if not isinstance(thread, dict):
                continue
            user_id = thread.get("userId")
            if user_id in legacy_ids or user_id is None:
                thread["userId"] = target_user_id
                changed = True
        if changed:
            self._save_data(data)

    def _has_real_user_step(self, thread: Dict[str, Any]) -> bool:
        steps = thread.get("steps") or []
        return any(
            isinstance(s, dict)
            and s.get("type") == "user"
            and bool((s.get("output") or "").strip())
            for s in steps
        )

    def _is_transient_step(self, step: Dict[str, Any]) -> bool:
        if not isinstance(step, dict):
            return False
        step_type = step.get("type")
        if step_type not in {"assistant_message", "assistant"}:
            return False
        output = step.get("output")
        if not isinstance(output, str):
            return False
        trimmed = output.strip()
        transient_prefixes = (
            "Calling ",
            "Submitting Kling job",
            "Kling error:",
        )
        return trimmed.startswith(transient_prefixes)

    def _filter_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        allowed_types = {"user", "assistant", "user_message", "assistant_message"}
        filtered: List[Dict[str, Any]] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            step_type = step.get("type")
            if step_type not in allowed_types:
                continue
            if self._is_transient_step(step):
                continue
            filtered.append(step)
        return filtered

    def _prune_empty_threads(self, data: Dict[str, Any]) -> None:
        to_delete = []
        for thread_id, thread in data.items():
            if not isinstance(thread, dict):
                to_delete.append(thread_id)
                continue
            steps = thread.get("steps") or []
            filtered_steps = self._filter_steps(list(steps))
            if len(filtered_steps) != len(steps):
                thread["steps"] = filtered_steps
            if not self._has_real_user_step(thread):
                to_delete.append(thread_id)
        if to_delete:
            for thread_id in to_delete:
                data.pop(thread_id, None)
            self._save_data(data)

    # --- Users ---
    async def get_user(self, identifier: str):
        resolved_identifier = identifier
        resolved_id = identifier or "user-1"
        created_at = _utc_now_iso()
        if isinstance(identifier, dict):
            resolved_identifier = identifier.get("identifier") or identifier.get("id") or ""
            resolved_id = identifier.get("id") or resolved_identifier or resolved_id
            created_at = identifier.get("createdAt") or created_at
        if resolved_identifier:
            self._last_identifier = resolved_identifier
        if resolved_id:
            self._last_user_id = resolved_id
        user_obj = PersistedUser(
            id=resolved_id,
            createdAt=created_at,
            identifier=resolved_identifier,
        )
        logger.info(
            "[JsonDataLayer] get_user identifier_type=%s resolved_identifier=%s",
            str(type(identifier)),
            resolved_identifier,
        )
        return user_obj

    async def create_user(self, user):
        # No real user persistence; mirror the default user.
        if isinstance(user, dict):
            created = await self.get_user(user)
            logger.info("[JsonDataLayer] create_user dict resolved_identifier=%s", created.identifier)
            return created
        identifier = getattr(user, "identifier", "")
        created = await self.get_user(identifier)
        logger.info("[JsonDataLayer] create_user resolved_identifier=%s", created.identifier)
        return created

    # --- Threads ---
    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        data = self._load_data()
        thread = data.get(thread_id)
        blocked_titles = {"provider", "model"}
        if isinstance(name, str):
            normalized = name.strip().lower()
        else:
            normalized = ""
        metadata_title_set = isinstance(metadata, dict) and bool(metadata.get("title_set"))
        if normalized in blocked_titles and not metadata_title_set:
            name = None
        if not thread:
            if (
                (name is None or name == "New Chat")
                and not user_id
                and not metadata
                and not tags
            ):
                return
            thread = {
                "id": thread_id,
                "createdAt": _utc_now_iso(),
                "name": name or "New Chat",
                "userId": user_id,
                "steps": [],
                "metadata": metadata or {},
                "tags": tags or [],
            }
        else:
            if name:
                thread["name"] = name
            if user_id:
                thread["userId"] = user_id
            if metadata:
                thread.setdefault("metadata", {}).update(metadata)
            if tags:
                thread["tags"] = tags
        data[thread_id] = thread
        self._save_data(data)

    async def list_threads(
        self, pagination: Pagination, filters: ThreadFilter
    ) -> PaginatedResponse:
        try:
            print(
                "[JsonDataLayer] list_threads called",
                {"pagination": pagination, "filters": filters},
            )
        except Exception:
            pass
        if not self._migrated_user_ids:
            target_user = self._last_identifier or self._last_user_id
            if isinstance(target_user, str) and target_user:
                self._migrate_thread_user_ids(target_user)
                self._migrated_user_ids = True
        data = self._load_data()
        threads = list(data.values())
        if isinstance(filters, ThreadFilter):
            user_id = getattr(filters, "userId", None)
            if user_id:
                allowed_ids = {user_id}
                if self._last_identifier:
                    allowed_ids.add(self._last_identifier)
                if self._last_user_id:
                    allowed_ids.add(self._last_user_id)
                threads = [t for t in threads if t.get("userId") in allowed_ids]
        threads.sort(key=lambda x: x.get("createdAt", ""), reverse=True)

        limit = pagination.first or 20
        page_items = threads[:limit]
        return PaginatedResponse(
            data=page_items,
            pageInfo=PageInfo(hasNextPage=len(threads) > limit, startCursor=None, endCursor=None),
        )

    async def get_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        data = self._load_data()
        thread = data.get(thread_id)
        if not isinstance(thread, dict):
            return None
        steps = thread.get("steps") or []
        filtered_steps = self._filter_steps(list(steps))
        if len(filtered_steps) != len(steps):
            thread["steps"] = filtered_steps
            data[thread_id] = thread
            self._save_data(data)
        if not self._has_real_user_step(thread):
            data.pop(thread_id, None)
            self._save_data(data)
            return None
        return thread

    async def delete_thread(self, thread_id: str):
        data = self._load_data()
        if thread_id in data:
            del data[thread_id]
            self._save_data(data)

    async def get_thread_author(self, thread_id: str) -> str:
        thread = await self.get_thread(thread_id)
        if thread:
            return thread.get("userId", "") or ""
        return ""

    # --- Steps ---
    async def create_step(self, step_dict: Dict[str, Any]):
        thread_id = step_dict.get("threadId")
        if not thread_id:
            return
        step_type = step_dict.get("type")
        is_user_step = step_type == "user" and bool((step_dict.get("output") or "").strip())
        data = self._load_data()
        thread = data.get(thread_id)
        if not thread and not is_user_step:
            return
        if not thread:
            thread = {
                "id": thread_id,
                "createdAt": _utc_now_iso(),
                "name": "New Chat",
                "userId": self._last_identifier or self._last_user_id,
                "steps": [],
                "metadata": {},
                "tags": [],
            }
        if thread and not is_user_step:
            steps = thread.get("steps") or []
            if not self._has_real_user_step(thread):
                return
        thread.setdefault("steps", []).append(step_dict)
        data[thread_id] = thread
        self._save_data(data)

    def _migrate_legacy_list(self, legacy: Any) -> Dict[str, Any]:
        if not isinstance(legacy, list) or not legacy:
            return {}

        thread_id = str(uuid.uuid4())
        steps: List[Dict[str, Any]] = []
        thread_created = None

        for item in legacy:
            if not isinstance(item, dict):
                continue
            ts = item.get("ts")
            if ts and thread_created is None:
                thread_created = ts

            user_msg = item.get("user")
            assistant_msg = item.get("assistant")
            created_at = ts or _utc_now_iso()

            if user_msg:
                steps.append(
                    {
                        "id": str(uuid.uuid4()),
                        "type": "user",
                        "threadId": thread_id,
                        "createdAt": created_at,
                        "output": user_msg,
                    }
                )

            if assistant_msg:
                steps.append(
                    {
                        "id": str(uuid.uuid4()),
                        "type": "assistant",
                        "threadId": thread_id,
                        "createdAt": created_at,
                        "output": assistant_msg,
                    }
                )

        thread_created = thread_created or _utc_now_iso()

        return {
            thread_id: {
                "id": thread_id,
                "createdAt": thread_created,
                "name": "Legacy Chat",
                "userId": "me",
                "steps": steps,
                "metadata": {},
                "tags": [],
            }
        }

    # --- Element/Step stubs (unused) ---
    async def get_element(self, thread_id, element_id):
        return None

    async def delete_element(self, element_id, thread_id=None):
        return None

    async def create_element(self, element):
        return None

    async def update_step(self, step_dict):
        return None

    async def delete_step(self, step_id):
        return None

    async def upsert_feedback(self, feedback):
        return ""

    async def delete_feedback(self, feedback_id):
        return False

    async def build_debug_url(self) -> str:
        return ""

    async def close(self) -> None:
        return None

    async def get_favorite_steps(self, user_id: str):
        return []


__all__ = [
    "JsonDataLayer",
    "HISTORY_PATH",
]