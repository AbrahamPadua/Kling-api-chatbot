import asyncio
import os
import uuid
from datetime import datetime
from typing import Dict, List, TypedDict, Optional, Any

import chainlit as cl
from chainlit.input_widget import Select, Slider

try:
    import chainlit.server as chainlit_server

    class _UserWrapper:
        def __init__(self, data: dict):
            self.identifier = data.get("identifier") or data.get("id") or ""

        def to_dict(self):
            return {"identifier": self.identifier}

    @chainlit_server.app.middleware("http")
    async def _ensure_user_identifier(request, call_next):
        user = getattr(request.state, "user", None)
        if isinstance(user, dict):
            request.state.user = _UserWrapper(user)
        return await call_next(request)
except Exception:
    pass

try:
    import chainlit.auth as cl_auth

    _orig_authenticate_user = cl_auth.authenticate_user

    async def _authenticate_user_patched(token: str):
        user = await _orig_authenticate_user(token)
        if isinstance(user, dict):
            identifier = user.get("identifier") or user.get("id") or ""
            display_name = user.get("display_name") if isinstance(user.get("display_name"), str) else None
            metadata = user.get("metadata") if isinstance(user.get("metadata"), dict) else {}
            return cl.User(identifier=identifier, display_name=display_name, metadata=metadata)
        return user

    cl_auth.authenticate_user = _authenticate_user_patched
except Exception:
    pass

from providers import (
    call_provider,
    call_kling_multi_image_to_video,
    call_kling_image_to_video,
    get_kling_task_result,
    ProviderError,
)
from json_datalayer import JsonDataLayer
from kling_flow import run_kling_flow, extract_files_from_response
from media_index import add_media_entry, render_media_page

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


class ProviderConfig(TypedDict):
    id: str
    models: List[str]
    default_temp: float


PROVIDERS: Dict[str, ProviderConfig] = {
    "Claude": {
        "id": "claude",
        "models": [
            "claude-3.5-sonnet-latest",
            "claude-3.5-sonnet",
            "claude-3.5-haiku-latest",
            "claude-3.5-haiku",
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3-haiku",
        ],
        "default_temp": 0.6,
    },
    "GPT": {
        "id": "gpt",
        "models": [
            "gpt-5.2",
            "gpt-5.1",
            "gpt-5.1-codex",
            "gpt-5",
            "gpt-5-codex",
            "gpt-5-chat-latest",
            "gpt-4.1",
            "gpt-4o",
            "o1",
            "o3",
            "gpt-5.1-codex-mini",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-4o-mini",
            "o1-mini",
            "o3-mini",
            "o4-mini",
            "codex-mini-latest",
        ],
        "default_temp": 0.7,
    },
    "Gemini": {
        "id": "gemini",
        "models": [
            "gemini-3-pro",
            "gemini-3-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash-live",
        ],
        "default_temp": 0.5,
    },
    "Kling": {
        "id": "kling",
        "models": [
            "kling-v1-6",
        ],
        "default_temp": 0.0,
    },
}


DATA_LAYER = JsonDataLayer()


@cl.data_layer  # type: ignore[arg-type]
def get_data_layer():
    try:
        print("[main] JsonDataLayer registered")
    except Exception:
        pass
    return DATA_LAYER


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    expected_user = os.getenv("CHAINLIT_AUTH_USERNAME")
    expected_pass = os.getenv("CHAINLIT_AUTH_PASSWORD")
    if not expected_user or not expected_pass:
        return None
    if username == expected_user and password == expected_pass:
        return cl.User(identifier=username)
    return None


def _reset_session() -> None:
    cl.user_session.set("provider_name", None)
    cl.user_session.set("provider_id", None)
    cl.user_session.set("model", None)
    cl.user_session.set("temperature", None)
    cl.user_session.set("system_prompt", DEFAULT_SYSTEM_PROMPT)
    cl.user_session.set("histories", {})  # per-provider histories


def _restore_session_from_metadata(metadata: Optional[Dict[str, object]]) -> bool:
    if not isinstance(metadata, dict):
        return False
    provider_name = metadata.get("provider_name")
    provider_id = metadata.get("provider_id")
    model = metadata.get("model")
    temperature = metadata.get("temperature")
    system_prompt = metadata.get("system_prompt")
    histories = metadata.get("histories")
    restored = False

    if isinstance(provider_name, str):
        cl.user_session.set("provider_name", provider_name)
        restored = True
    if isinstance(provider_id, str):
        cl.user_session.set("provider_id", provider_id)
        restored = True
    if isinstance(model, str):
        cl.user_session.set("model", model)
        restored = True
    if isinstance(temperature, (int, float)):
        cl.user_session.set("temperature", float(temperature))
        restored = True
    if isinstance(system_prompt, str):
        cl.user_session.set("system_prompt", system_prompt)
        restored = True
    if isinstance(histories, dict):
        cl.user_session.set("histories", histories)
        restored = True

    return restored


def _rebuild_histories_from_thread(thread: Optional[Dict[str, object]]) -> bool:
    if not isinstance(thread, dict):
        return False
    metadata = thread.get("metadata")
    provider_id = None
    if isinstance(metadata, dict):
        provider_id = metadata.get("provider_id")
    if not isinstance(provider_id, str) or not provider_id:
        provider_id = cl.user_session.get("provider_id")
    if not isinstance(provider_id, str) or not provider_id:
        return False

    steps = thread.get("steps")
    if not isinstance(steps, list):
        return False

    ordered_steps: List[Dict[str, Any]] = [s for s in steps if isinstance(s, dict)]
    ordered_steps.sort(key=lambda s: str(s.get("createdAt") or ""))

    def _build_from_types(type_map: Dict[str, str]) -> List[Dict[str, str]]:
        conv: List[Dict[str, str]] = []
        for step in ordered_steps:
            step_type = step.get("type")
            role = type_map.get(step_type) if isinstance(step_type, str) else None
            if not role:
                continue
            output = step.get("output")
            if isinstance(output, str) and output.strip():
                conv.append({"role": role, "content": output.strip()})
        return conv

    conversation = _build_from_types({"user": "user", "assistant": "assistant"})
    if not conversation:
        conversation = _build_from_types({"user_message": "user", "assistant_message": "assistant"})
    if not conversation:
        return False

    histories = cl.user_session.get("histories") or {}
    if not isinstance(histories, dict):
        histories = {}
    histories[provider_id] = conversation
    cl.user_session.set("histories", histories)
    return True


def _thread_title_set(thread: Optional[Dict[str, object]]) -> bool:
    if not isinstance(thread, dict):
        return False
    metadata = thread.get("metadata")
    if isinstance(metadata, dict):
        return bool(metadata.get("title_set"))
    return False


def _make_thread_title(text: str, max_len: int = 50) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return "New Chat"
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rstrip() + "…"


def _sanitize_title(text: str, max_len: int = 50) -> str:
    cleaned = " ".join(text.strip().strip("'\"").split())
    if not cleaned:
        return "New Chat"
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rstrip() + "…"


async def _generate_title_from_prompt(prompt: str) -> str:
    fallback = _make_thread_title(prompt, max_len=50)
    if not prompt.strip():
        return fallback
    system_prompt = (
        "You generate short chat titles. "
        "Return only the title, no quotes, no punctuation or commentary. "
        "Title must be under 50 characters."
    )
    try:
        title = await asyncio.to_thread(
            call_provider,
            "gpt",
            "gpt-5.1",
            [{"role": "user", "content": prompt}],
            0.2,
            system_prompt,
            32,
        )
        if not isinstance(title, str):
            return fallback
        return _sanitize_title(title, max_len=50)
    except Exception:
        return fallback


def _current_user_identifier() -> Optional[str]:
    user = cl.user_session.get("user")
    if isinstance(user, dict):
        return user.get("identifier") or user.get("id")
    if hasattr(user, "identifier"):
        return getattr(user, "identifier")
    return None


def _current_user_id() -> Optional[str]:
    user = cl.user_session.get("user")
    if isinstance(user, dict):
        return user.get("id")
    if hasattr(user, "id"):
        return getattr(user, "id")
    return None


def _active_thread_id() -> Optional[str]:
    thread_id = cl.user_session.get("thread_id")
    if isinstance(thread_id, str) and thread_id:
        return thread_id
    ctx = getattr(cl, "context", None)
    session = getattr(ctx, "session", None) if ctx else None
    thread_id = getattr(session, "thread_id", None) if session else None
    if isinstance(thread_id, str) and thread_id:
        return thread_id
    return None


def _extract_text_input(response) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        if isinstance(response.get("content"), str):
            return response.get("content", "")
        if isinstance(response.get("output"), str):
            return response.get("output", "")
    if not isinstance(response, dict):
        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content
        output = getattr(response, "output", None)
        if isinstance(output, str):
            return output
    return ""


async def _ask_text(prompt: str, default: Optional[str] = None, timeout: int = 600) -> str:
    reply = await cl.AskUserMessage(content=prompt, timeout=timeout).send()
    text = _extract_text_input(reply).strip()
    if not text and default is not None:
        return default
    return text


async def _ensure_thread_id() -> str:
    thread_id = _active_thread_id()
    if thread_id:
        return thread_id
    thread_id = str(uuid.uuid4())
    cl.user_session.set("thread_id", thread_id)
    return thread_id


def _history_for(provider_id: str) -> List[Dict[str, str]]:
    histories = cl.user_session.get("histories") or {}
    if provider_id not in histories:
        histories[provider_id] = []
        cl.user_session.set("histories", histories)
    return histories[provider_id]


def _set_provider(provider_name: str) -> None:
    if provider_name not in PROVIDERS:
        raise ProviderError(f"Unknown provider: {provider_name}")
    cfg = PROVIDERS[provider_name]
    cl.user_session.set("provider_name", provider_name)
    cl.user_session.set("provider_id", cfg["id"])
    cl.user_session.set("temperature", cfg["default_temp"])
    cl.user_session.set("model", None)


def _set_model(model: str) -> None:
    provider_name = cl.user_session.get("provider_name")
    if not provider_name:
        raise ProviderError("Pick a provider before choosing a model.")
    models = PROVIDERS[provider_name]["models"]
    if model not in models:
        raise ProviderError(f"Unknown model for {provider_name}: {model}")
    cl.user_session.set("model", model)

def _ensure_default_model(provider_name: str) -> Optional[str]:
    models = PROVIDERS[provider_name]["models"]
    if len(models) == 1:
        cl.user_session.set("model", models[0])
        return models[0]
    return None


def _set_temperature(value: float) -> None:
    cl.user_session.set("temperature", value)


async def _send_chat_settings() -> None:
    provider_name = cl.user_session.get("provider_name")
    if not isinstance(provider_name, str) or provider_name not in PROVIDERS:
        provider_name = list(PROVIDERS.keys())[0]
    models = PROVIDERS[provider_name]["models"]
    current_model = cl.user_session.get("model")
    if not isinstance(current_model, str) or current_model not in models:
        current_model = models[0] if models else ""
    temperature = cl.user_session.get("temperature")
    if not isinstance(temperature, (int, float)):
        temperature = PROVIDERS[provider_name]["default_temp"]

    provider_options = list(PROVIDERS.keys())
    provider_index = provider_options.index(provider_name) if provider_name in provider_options else 0
    model_index = models.index(current_model) if current_model in models else 0

    await cl.ChatSettings(
        [
            Select(
                id="Provider",
                label="Provider",
                values=provider_options,
                initial_index=provider_index,
            ),
            Select(
                id="Model",
                label=f"Model ({provider_name})",
                values=models,
                initial_index=model_index,
            ),
            Slider(
                id="Temperature",
                label="Temperature",
                initial=float(temperature),
                min=0,
                max=2,
                step=0.1,
            ),
        ]
    ).send()


async def _ask_provider() -> None:
    actions = [cl.Action(name="provider", label=name, payload={"provider": name}) for name in PROVIDERS.keys()]
    content = "Pick a provider to begin:\n" + "\n".join(f"- {name}" for name in PROVIDERS.keys())
    await cl.Message(content=content, actions=actions, author="system").send()


async def _ask_model(provider_name: str) -> None:
    models = PROVIDERS[provider_name]["models"]
    actions = [cl.Action(name="model", label=m, payload={"model": m}) for m in models]
    content = "Select a model from the options below:\n" + "\n".join(f"- {m}" for m in models)
    await cl.Message(content=content, actions=actions if actions else None, author="system").send()


async def _parse_command(message: cl.Message) -> bool:
    text = message.content.strip()
    lowered = text.lower()
    if lowered.startswith("/new"):
        _reset_session()
        await _ask_provider()
        return True
    if lowered.startswith("/providers") or lowered.startswith("/picker"):
        await _ask_provider()
        provider_name = cl.user_session.get("provider_name")
        if provider_name:
            await _ask_model(provider_name)
        return True
    if lowered.startswith("/models"):
        provider_name = cl.user_session.get("provider_name")
        if provider_name:
            await _ask_model(provider_name)
        else:
            await _ask_provider()
        return True
    if lowered.startswith("/temp"):
        parts = text.split()
        if len(parts) >= 2:
            try:
                val = float(parts[1])
                _set_temperature(val)
                await cl.Message(content=f"Temperature set to {val:.2f}", author="system").send()
            except ValueError:
                await cl.Message(content="Temperature must be a number.", author="system").send()
        else:
            await cl.Message(content="Usage: /temp 0.7", author="system").send()
        return True
    if lowered.startswith("/model"):
        parts = text.split(maxsplit=1)
        if len(parts) == 2:
            try:
                _set_model(parts[1].strip())
                await cl.Message(content=f"Model set to {parts[1].strip()}", author="system").send()
            except ProviderError as exc:
                await cl.Message(content=str(exc), author="system").send()
                provider_name = cl.user_session.get("provider_name")
                if provider_name:
                    await _ask_model(provider_name)
                else:
                    await _ask_provider()
        else:
            await cl.Message(content="Usage: /model your-model-name", author="system").send()
            provider_name = cl.user_session.get("provider_name")
            if provider_name:
                await _ask_model(provider_name)
            else:
                await _ask_provider()
        return True
    if lowered.startswith("/media"):
        parts = text.split(maxsplit=1)
        page = 1
        if len(parts) == 2:
            try:
                page = int(parts[1].strip())
            except ValueError:
                page = 1
        await cl.Message(content=render_media_page(page), author="system").send()
        return True
    if lowered.startswith("/kling_task"):
        parts = text.split(maxsplit=1)
        task_id = parts[1].strip() if len(parts) == 2 else ""
        if not task_id:
            task_id = await _ask_text("Enter the Kling task_id:", "", 600)
        if not task_id:
            await cl.Message(content="Kling task_id is required.", author="system").send()
            return True
        status_msg = await cl.Message(content="Checking Kling task status...", author="system").send()
        try:
            result = await asyncio.to_thread(get_kling_task_result, task_id)
            status = result.get("status")
            video_url = result.get("video_url")
            if video_url:
                thread_id = await _ensure_thread_id()
                add_media_entry(video_url, "video", thread_id, "Kling")
                reply = f"Kling task {task_id} ready: {video_url}"
            else:
                reply = f"Kling task {task_id} status: {status or 'unknown'}"
            if status_msg:
                status_msg.content = reply
                await status_msg.update()
        except ProviderError as exc:
            if status_msg:
                status_msg.content = f"Kling task lookup failed: {exc}"
                await status_msg.update()
        return True
    if lowered.startswith("/provider"):
        parts = text.split(maxsplit=1)
        if len(parts) == 2:
            try:
                _set_provider(parts[1].strip())
                await cl.Message(content=f"Provider set to {parts[1].strip()}", author="system").send()
                await _ask_model(parts[1].strip())
            except ProviderError as exc:
                await cl.Message(content=str(exc), author="system").send()
                await _ask_provider()
        else:
            await cl.Message(content="Usage: /provider Claude|GPT|Gemini|Kling", author="system").send()
            await _ask_provider()
        return True
    return False


@cl.on_chat_start
async def on_start():
    thread_id = _active_thread_id()
    if thread_id:
        thread = await DATA_LAYER.get_thread(thread_id)
        if isinstance(thread, dict) and thread.get("steps"):
            steps = thread.get("steps") or []
            has_user_step = any(
                isinstance(step, dict) and step.get("type") in {"user", "user_message"}
                for step in steps
            )
            if has_user_step:
                metadata = thread.get("metadata")
                _restore_session_from_metadata(metadata)
                _rebuild_histories_from_thread(thread)
                return

    _reset_session()
    await _send_chat_settings()
    await cl.Message(
        content=(
            "Welcome! This chat now runs in Chainlit.\n"
            "- Use /new to reset the chat.\n"
            "- Use /temp 0.7 to set temperature.\n"
            "- Use /model model-name to set a model manually.\n"
            "- Use /provider ProviderName to switch providers.\n"
            "Pick a provider below to start."
        ),
        author="system",
    ).send()
    await _ask_provider()


@cl.on_chat_resume
async def on_resume(thread):
    thread_id = None
    if isinstance(thread, dict):
        thread_id = thread.get("id")
    if isinstance(thread_id, str) and thread_id:
        cl.user_session.set("thread_id", thread_id)

    if isinstance(thread, dict):
        metadata = thread.get("metadata")
        restored = _restore_session_from_metadata(metadata)
        rebuilt = _rebuild_histories_from_thread(thread)
        if restored or rebuilt:
            await _send_chat_settings()
            return

    await _send_chat_settings()
    await _ask_provider()


@cl.on_settings_update
async def on_settings_update(settings):
    provider_name = settings.get("Provider") if isinstance(settings, dict) else None
    model_name = settings.get("Model") if isinstance(settings, dict) else None
    temperature = settings.get("Temperature") if isinstance(settings, dict) else None

    if not isinstance(provider_name, str) or provider_name not in PROVIDERS:
        provider_name = cl.user_session.get("provider_name")

    if isinstance(provider_name, str) and provider_name in PROVIDERS:
        _set_provider(provider_name)
        models = PROVIDERS[provider_name]["models"]
        if not isinstance(model_name, str) or model_name not in models:
            model_name = _ensure_default_model(provider_name)
        if isinstance(model_name, str) and model_name in models:
            _set_model(model_name)

    if isinstance(temperature, (int, float)):
        _set_temperature(float(temperature))

    await _send_chat_settings()


@cl.action_callback("provider")
async def provider_selected(action: cl.Action):
    provider_val = None
    if hasattr(action, "payload") and isinstance(action.payload, dict):
        provider_val = action.payload.get("provider")
    if provider_val is None:
        await cl.Message(content="Invalid provider selection. Please pick again.", author="system").send()
        await _ask_provider()
        return
    try:
        _set_provider(provider_val)
        await cl.Message(content=f"Provider set to {provider_val}.", author="system").send()
        await _send_chat_settings()
        default_model = _ensure_default_model(provider_val)
        if provider_val == "Kling" and default_model:
            await run_kling_flow(
                ask_text=_ask_text,
                ensure_thread_id=_ensure_thread_id,
                data_layer=DATA_LAYER,
                add_media_entry=add_media_entry,
                call_kling_multi_image_to_video=call_kling_multi_image_to_video,
                call_kling_image_to_video=call_kling_image_to_video,
                extract_text_input=_extract_text_input,
                generate_title=_generate_title_from_prompt,
                initial_text="",
                initial_files=None,
            )
            return
        await _ask_model(provider_val)
    except ProviderError as exc:
        await cl.Message(content=str(exc), author="system").send()
        await _ask_provider()


@cl.action_callback("model")
async def model_selected(action: cl.Action):
    model_val = None
    if hasattr(action, "payload") and isinstance(action.payload, dict):
        model_val = action.payload.get("model")
    if model_val is None:
        await cl.Message(content="Invalid model selection. Please pick again.", author="system").send()
        provider_name = cl.user_session.get("provider_name")
        if provider_name:
            await _ask_model(provider_name)
        else:
            await _ask_provider()
        return
    try:
        _set_model(model_val)
        temp = cl.user_session.get("temperature") or 0.7
        await cl.Message(content=f"Model set to {model_val}. Temperature {temp:.2f}. Use /temp X to change.", author="system").send()
        await _send_chat_settings()
    except ProviderError as exc:
        await cl.Message(content=str(exc), author="system").send()
        provider_name = cl.user_session.get("provider_name")
        if provider_name:
            await _ask_model(provider_name)
        else:
            await _ask_provider()


@cl.on_message
async def on_message(message: cl.Message):
    text = message.content.strip()
    if not text:
        return

    if await _parse_command(message):
        return

    provider_name = cl.user_session.get("provider_name")
    provider_id = cl.user_session.get("provider_id")
    model = cl.user_session.get("model")
    temperature = cl.user_session.get("temperature") or 0.7
    system_prompt = cl.user_session.get("system_prompt") or DEFAULT_SYSTEM_PROMPT

    if not provider_name or not provider_id:
        await cl.Message(content="Please pick a provider first.", author="system").send()
        await _ask_provider()
        return
    if not model:
        default_model = _ensure_default_model(provider_name)
        if provider_id == "kling" and default_model:
            await run_kling_flow(
                ask_text=_ask_text,
                ensure_thread_id=_ensure_thread_id,
                data_layer=DATA_LAYER,
                add_media_entry=add_media_entry,
                call_kling_multi_image_to_video=call_kling_multi_image_to_video,
                call_kling_image_to_video=call_kling_image_to_video,
                extract_text_input=_extract_text_input,
                generate_title=_generate_title_from_prompt,
                initial_text=text,
                initial_files=extract_files_from_response(message),
            )
            return
        await cl.Message(content="Please pick a model.", author="system").send()
        await _ask_model(provider_name)
        return

    if provider_id == "kling":
        initial_files = extract_files_from_response(message)
        await run_kling_flow(
            ask_text=_ask_text,
            ensure_thread_id=_ensure_thread_id,
            data_layer=DATA_LAYER,
            add_media_entry=add_media_entry,
            call_kling_multi_image_to_video=call_kling_multi_image_to_video,
            call_kling_image_to_video=call_kling_image_to_video,
            extract_text_input=_extract_text_input,
            generate_title=_generate_title_from_prompt,
            initial_text=text,
            initial_files=initial_files,
        )
        return

    history = _history_for(provider_id)
    history.append({"role": "user", "content": text})
    thread_id = await _ensure_thread_id()
    thread = await DATA_LAYER.get_thread(thread_id)
    if not _thread_title_set(thread):
        title = await _generate_title_from_prompt(text)
        await DATA_LAYER.update_thread(
            thread_id=thread_id,
            name=title,
            user_id=_current_user_id() or _current_user_identifier(),
            metadata={"title_set": True},
        )
    else:
        await DATA_LAYER.update_thread(
            thread_id=thread_id,
            user_id=_current_user_id() or _current_user_identifier(),
        )

    # Save user step to data layer
    await DATA_LAYER.create_step(
        {
            "id": str(uuid.uuid4()),
            "type": "user",
            "threadId": thread_id,
            "createdAt": datetime.utcnow().isoformat() + "Z",
            "output": text,
        }
    )

    status = await cl.Message(content=f"Calling {provider_name} ({model})...", author="system").send()
    try:
        reply = await asyncio.to_thread(
            call_provider,
            provider_id,
            model,
            history,
            float(temperature),
            system_prompt,
            2048,
        )
        if not reply:
            reply = "(No content received from the provider.)"
        history.append({"role": "assistant", "content": reply})

        await DATA_LAYER.create_step(
            {
                "id": str(uuid.uuid4()),
                "type": "assistant",
                "threadId": thread_id,
                "createdAt": datetime.utcnow().isoformat() + "Z",
                "output": reply,
            }
        )

        if status:
            status.content = reply
            status.author = provider_name
            await status.update()
    except ProviderError as exc:
        history.pop()  # remove user turn on failure
        if status:
            status.content = f"Error: {exc}"
            status.author = "system"
            await status.update()
    except Exception as exc:  # catch-all to surface unexpected errors
        history.pop()
        if status:
            status.content = f"Unexpected error: {exc}"
            status.author = "system"
            await status.update()
