import asyncio
import base64
import re
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any

import chainlit as cl
import requests

from providers import ProviderError

TextAskFunc = Callable[[str, Optional[str], int], Any]
EnsureThreadFunc = Callable[[], Any]
AddMediaFunc = Callable[[str, str, Optional[str], str], None]
CallImageToVideoFunc = Callable[..., str]
TitleFunc = Callable[[str], Any]


def parse_image_inputs(raw: str) -> List[str]:
    text = raw.strip()
    if not text:
        return []
    if "http://" in text or "https://" in text:
        urls = re.findall(r"https?://[^\s,]+", text)
        if urls:
            return urls
    parts = re.split(r"[\n,;]+", text)
    return [p.strip() for p in parts if p.strip()]


def extract_files_from_response(response) -> List[object]:
    if response is None:
        return []
    if isinstance(response, list):
        return response
    if isinstance(response, dict):
        files = response.get("files") or response.get("elements") or response.get("attachments")
        if isinstance(files, list):
            return files
    if hasattr(response, "files"):
        files = getattr(response, "files")
        if isinstance(files, list):
            return files
    if hasattr(response, "elements"):
        elems = getattr(response, "elements")
        if isinstance(elems, list):
            return elems
    if hasattr(response, "attachments"):
        atts = getattr(response, "attachments")
        if isinstance(atts, list):
            return atts
    return []


def strip_command(text: str, command: str) -> str:
    lowered = text.strip().lower()
    if lowered.startswith(command):
        return text.strip()[len(command):].strip()
    return ""


def get_file_path(file_obj: object) -> Optional[str]:
    if hasattr(file_obj, "path"):
        return getattr(file_obj, "path")
    if isinstance(file_obj, dict):
        return (
            file_obj.get("path")
            or file_obj.get("filePath")
            or file_obj.get("filepath")
            or file_obj.get("path_on_disk")
            or file_obj.get("pathOnDisk")
            or file_obj.get("local_path")
        )
    return None


def get_file_url(file_obj: object) -> Optional[str]:
    if isinstance(file_obj, dict):
        return file_obj.get("url") or file_obj.get("download_url") or file_obj.get("contentUrl")
    if hasattr(file_obj, "url"):
        return getattr(file_obj, "url")
    if hasattr(file_obj, "download_url"):
        return getattr(file_obj, "download_url")
    return None


def download_image_as_base64(url: str, max_bytes: int = 10 * 1024 * 1024) -> str:
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise ProviderError(f"Failed to download image URL: {url} ({exc})") from exc

    content_type = resp.headers.get("Content-Type", "")
    if content_type and not content_type.startswith("image/"):
        raise ProviderError(f"URL did not return an image content-type: {url} ({content_type})")

    total = 0
    chunks: List[bytes] = []
    for chunk in resp.iter_content(chunk_size=1024 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            raise ProviderError(f"Image exceeds 10MB limit: {url}")
        chunks.append(chunk)

    data = b"".join(chunks)
    if not data:
        raise ProviderError(f"Empty image data from URL: {url}")
    return base64.b64encode(data).decode("utf-8")


def read_file_as_base64(file_path: str, max_bytes: int = 10 * 1024 * 1024) -> str:
    try:
        with open(file_path, "rb") as f:
            data = f.read(max_bytes + 1)
    except Exception as exc:
        raise ProviderError(f"Failed to read file: {file_path} ({exc})") from exc
    if len(data) > max_bytes:
        raise ProviderError(f"File exceeds 10MB limit: {file_path}")
    if not data:
        raise ProviderError(f"Empty file: {file_path}")
    return base64.b64encode(data).decode("utf-8")


async def run_kling_multi_image_flow(
    *,
    ask_text: TextAskFunc,
    ensure_thread_id: EnsureThreadFunc,
    data_layer,
    add_media_entry: AddMediaFunc,
    call_kling_multi_image_to_video: Callable[..., str],
    extract_text_input: Callable[[object], str],
    generate_title: Optional[TitleFunc] = None,
    initial_text: str = "",
    initial_files: Optional[List[object]] = None,
) -> None:
    async def ask_choice(prompt: str, options: List[str], default: Optional[str] = None) -> str:
        actions = [cl.Action(name="choice", label=opt, payload={"value": opt}) for opt in options]
        response = await cl.AskActionMessage(content=prompt, actions=actions, timeout=600).send()
        if response is None:
            return default or options[0]
        if isinstance(response, dict):
            payload = response.get("payload")
            if isinstance(payload, dict):
                value = payload.get("value")
                if isinstance(value, str):
                    return value
        if hasattr(response, "get"):
            try:
                payload = response.get("payload")
                if isinstance(payload, dict):
                    value = payload.get("value")
                    if isinstance(value, str):
                        return value
            except Exception:
                pass
        value = getattr(response, "value", None)
        if isinstance(value, str):
            return value
        return default or options[0]

    await cl.Message(
        content=(
            "Kling Multi-Image → Video setup.\n"
            "Provide 1–4 images (upload, URL, or Base64) for reference and a prompt.\n"
            "If using Base64, paste the raw string only (no data:image/... prefix)."
        ),
        author="system",
    ).send()

    prompt = await ask_text("Enter your Kling prompt (required):", None, 600)
    if not prompt:
        await cl.Message(content="Prompt is required. Please run /kling again.", author="system").send()
        return

    negative_choice = await ask_choice(
        "Negative prompt (optional):",
        ["Skip", "Enter text"],
        default="Skip",
    )
    negative_prompt = ""
    if negative_choice == "Enter text":
        negative_prompt = await ask_text("Enter negative prompt:", "", 600)

    mode = await ask_choice("Select mode:", ["std", "pro"], default="std")
    aspect_ratio = await ask_choice("Select aspect ratio:", ["16:9", "9:16", "1:1"], default="16:9")
    duration = await ask_choice("Select duration:", ["5", "10"], default="5")

    uploaded_files = list(initial_files or [])

    upload_choice = await ask_choice(
        "Upload images now?",
        ["Upload", "Skip"],
        default="Upload",
    )
    if upload_choice == "Upload":
        new_files = await cl.AskFileMessage(
            content="Upload 1–4 images (.jpg/.jpeg/.png).",
            accept=["image/png", "image/jpeg"],
            max_files=4,
            timeout=600,
        ).send()
        uploaded_files.extend(list(new_files or []))

    text_input = ""
    if not uploaded_files:
        text_input = await ask_text(
            "Paste 1–4 image URLs or Base64 strings (optional if uploading files).",
            "",
            600,
        )
    image_items: List[str] = []
    image_items.extend(parse_image_inputs(initial_text))
    image_items.extend(parse_image_inputs(text_input))

    if not image_items and not uploaded_files:
        await cl.Message(content="At least one image is required. Please run /kling again.", author="system").send()
        return

    if len(uploaded_files) > 4:
        uploaded_files = uploaded_files[:4]
    if len(image_items) > 4:
        image_items = image_items[:4]

    remaining = 4 - len(uploaded_files)
    if remaining < len(image_items):
        image_items = image_items[:remaining]

    add_first_end = await ask_choice(
        "Mark first and last image as first_frame/end_frame?",
        ["Yes", "No"],
        default="Yes",
    )
    mark_first_end = add_first_end == "Yes"

    convert_urls = await ask_choice(
        "Convert image URLs to Base64 for compatibility?",
        ["Yes", "No"],
        default="Yes",
    )
    do_convert = convert_urls == "Yes"

    image_list: List[Dict[str, str]] = []
    for file_obj in uploaded_files or []:
        file_path = get_file_path(file_obj)
        try:
            if file_path:
                encoded = await asyncio.to_thread(read_file_as_base64, file_path)
            else:
                file_url = get_file_url(file_obj)
                if not file_url:
                    await cl.Message(content="Uploaded file missing path or URL metadata.", author="system").send()
                    return
                encoded = await asyncio.to_thread(download_image_as_base64, file_url)
            image_list.append({"image": encoded})
        except ProviderError as exc:
            await cl.Message(content=f"File upload conversion failed: {exc}", author="system").send()
            return

    for idx, val in enumerate(image_items):
        entry: Dict[str, str] = {}
        is_url = val.startswith("http://") or val.startswith("https://")
        if is_url and do_convert:
            try:
                encoded = await asyncio.to_thread(download_image_as_base64, val)
                entry["image"] = encoded
            except ProviderError as exc:
                await cl.Message(content=f"Image download/encode failed: {exc}", author="system").send()
                return
        elif is_url:
            entry["image_url"] = val
        else:
            entry["image"] = val

        total_images = len(image_items) + len(uploaded_files or [])
        if mark_first_end and total_images >= 2:
            if idx == 0 and not (uploaded_files or []):
                entry["type"] = "first_frame"
            elif idx == len(image_items) - 1:
                entry["type"] = "end_frame"

        image_list.append(entry)

    if mark_first_end and len(image_list) >= 2:
        image_list[0]["type"] = "first_frame"
        image_list[-1]["type"] = "end_frame"

    thread_id = await ensure_thread_id()
    thread = await data_layer.get_thread(thread_id)
    metadata = thread.get("metadata") if isinstance(thread, dict) else None
    should_rename = not (isinstance(metadata, dict) and metadata.get("title_set"))
    title_value = "Kling task"
    if should_rename and generate_title and isinstance(prompt, str) and prompt.strip():
        try:
            title_value = await generate_title(prompt)
        except Exception:
            title_value = prompt.strip() or title_value
    user_summary = (
        "Kling Multi-Image → Video request\n"
        f"Prompt: {prompt}\n"
        f"Mode: {mode}, Aspect: {aspect_ratio}, Duration: {duration}\n"
        f"Images: {len(image_list)}"
    )
    if should_rename:
        await data_layer.update_thread(
            thread_id=thread_id,
            name=title_value,
            user_id=None,
            metadata={"title_set": True},
        )
    await data_layer.create_step(
        {
            "id": str(uuid.uuid4()),
            "type": "user",
            "threadId": thread_id,
            "createdAt": datetime.utcnow().isoformat() + "Z",
            "output": user_summary,
        }
    )

    status = await cl.Message(content="Submitting Kling job and polling...", author="system").send()
    try:
        video_url = await asyncio.to_thread(
            call_kling_multi_image_to_video,
            prompt,
            image_list,
            negative_prompt or None,
            "kling-v1-6",
            mode,
            aspect_ratio,
            duration,
        )
        add_media_entry(video_url, "video", thread_id, "Kling")
        reply = f"Kling video ready: {video_url}"
        await data_layer.create_step(
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
            status.author = "Kling"
            await status.update()
    except ProviderError as exc:
        if status:
            status.content = f"Kling error: {exc}"
            status.author = "system"
            await status.update()
    except Exception as exc:
        if status:
            status.content = f"Unexpected Kling error: {exc}"
            status.author = "system"
            await status.update()


async def run_kling_image_to_video_flow(
    *,
    ask_text: TextAskFunc,
    ensure_thread_id: EnsureThreadFunc,
    data_layer,
    add_media_entry: AddMediaFunc,
    call_kling_image_to_video: CallImageToVideoFunc,
    generate_title: Optional[TitleFunc] = None,
    initial_text: str = "",
    initial_files: Optional[List[object]] = None,
) -> None:
    async def ask_choice(prompt: str, options: List[str], default: Optional[str] = None) -> str:
        actions = [cl.Action(name="choice", label=opt, payload={"value": opt}) for opt in options]
        response = await cl.AskActionMessage(content=prompt, actions=actions, timeout=600).send()
        if response is None:
            return default or options[0]
        if isinstance(response, dict):
            payload = response.get("payload")
            if isinstance(payload, dict):
                value = payload.get("value")
                if isinstance(value, str):
                    return value
        if hasattr(response, "get"):
            try:
                payload = response.get("payload")
                if isinstance(payload, dict):
                    value = payload.get("value")
                    if isinstance(value, str):
                        return value
            except Exception:
                pass
        value = getattr(response, "value", None)
        if isinstance(value, str):
            return value
        return default or options[0]

    await cl.Message(
        content=(
            "Kling Image → Video setup.\n"
            "Provide 1 image (upload, URL, or Base64). Optional end frame is supported.\n"
            "If using Base64, paste the raw string only (no data:image/... prefix)."
        ),
        author="system",
    ).send()

    prompt = await ask_text("Enter your Kling prompt (optional):", "", 600)
    negative_choice = await ask_choice(
        "Negative prompt (optional):",
        ["Skip", "Enter text"],
        default="Skip",
    )
    negative_prompt = ""
    if negative_choice == "Enter text":
        negative_prompt = await ask_text("Enter negative prompt:", "", 600)

    mode = await ask_choice("Select mode:", ["std", "pro"], default="std")
    duration = await ask_choice("Select duration:", ["5", "10"], default="5")

    upload_choice = await ask_choice(
        "Upload a reference image now?",
        ["Upload", "Skip"],
        default="Upload",
    )
    uploaded_files = list(initial_files or [])
    if upload_choice == "Upload":
        new_files = await cl.AskFileMessage(
            content="Upload 1 image (.jpg/.jpeg/.png).",
            accept=["image/png", "image/jpeg"],
            max_files=1,
            timeout=600,
        ).send()
        uploaded_files.extend(list(new_files or []))

    text_input = await ask_text(
        "Paste 1 image URL or Base64 (optional if uploading files).",
        "",
        600,
    )
    image_items = []
    image_items.extend(parse_image_inputs(initial_text))
    image_items.extend(parse_image_inputs(text_input))

    image_value: Optional[str] = None
    if uploaded_files:
        file_obj = uploaded_files[0]
        file_path = get_file_path(file_obj)
        try:
            if file_path:
                image_value = await asyncio.to_thread(read_file_as_base64, file_path)
            else:
                file_url = get_file_url(file_obj)
                if file_url:
                    image_value = await asyncio.to_thread(download_image_as_base64, file_url)
        except ProviderError as exc:
            await cl.Message(content=f"File upload conversion failed: {exc}", author="system").send()
            return

    if not image_value and image_items:
        image_value = image_items[0]

    if not image_value:
        await cl.Message(content="An image is required. Please try again.", author="system").send()
        return

    end_choice = await ask_choice(
        "Add an end frame (image_tail)?",
        ["No", "Yes"],
        default="No",
    )
    image_tail_value: Optional[str] = None
    if end_choice == "Yes":
        tail_upload = await ask_choice("Upload end frame?", ["Upload", "Skip"], default="Upload")
        tail_files: List[object] = []
        if tail_upload == "Upload":
            new_tail = await cl.AskFileMessage(
                content="Upload end frame image (.jpg/.jpeg/.png).",
                accept=["image/png", "image/jpeg"],
                max_files=1,
                timeout=600,
            ).send()
            tail_files.extend(list(new_tail or []))
        tail_text = await ask_text(
            "Paste end frame URL or Base64 (optional if uploading).",
            "",
            600,
        )
        tail_items = parse_image_inputs(tail_text)
        if tail_files:
            file_obj = tail_files[0]
            file_path = get_file_path(file_obj)
            try:
                if file_path:
                    image_tail_value = await asyncio.to_thread(read_file_as_base64, file_path)
                else:
                    file_url = get_file_url(file_obj)
                    if file_url:
                        image_tail_value = await asyncio.to_thread(download_image_as_base64, file_url)
            except ProviderError as exc:
                await cl.Message(content=f"End frame conversion failed: {exc}", author="system").send()
                return
        if not image_tail_value and tail_items:
            image_tail_value = tail_items[0]

    thread_id = await ensure_thread_id()
    thread = await data_layer.get_thread(thread_id)
    metadata = thread.get("metadata") if isinstance(thread, dict) else None
    should_rename = not (isinstance(metadata, dict) and metadata.get("title_set"))
    title_value = "Kling task"
    if should_rename and generate_title and isinstance(prompt, str) and prompt.strip():
        try:
            title_value = await generate_title(prompt)
        except Exception:
            title_value = prompt.strip() or title_value
    user_summary = (
        "Kling Image → Video request\n"
        f"Prompt: {prompt or '(none)'}\n"
        f"Mode: {mode}, Duration: {duration}\n"
        f"End frame: {'yes' if image_tail_value else 'no'}"
    )
    if should_rename:
        await data_layer.update_thread(
            thread_id=thread_id,
            name=title_value,
            user_id=None,
            metadata={"title_set": True},
        )
    await data_layer.create_step(
        {
            "id": str(uuid.uuid4()),
            "type": "user",
            "threadId": thread_id,
            "createdAt": datetime.utcnow().isoformat() + "Z",
            "output": user_summary,
        }
    )

    status = await cl.Message(content="Submitting Kling job and polling...", author="system").send()
    try:
        video_url = await asyncio.to_thread(
            call_kling_image_to_video,
            image_value,
            prompt or None,
            image_tail_value,
            negative_prompt or None,
            "kling-v1-6",
            mode,
            duration,
        )
        add_media_entry(video_url, "video", thread_id, "Kling")
        reply = f"Kling video ready: {video_url}"
        await data_layer.create_step(
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
            status.author = "Kling"
            await status.update()
    except ProviderError as exc:
        if status:
            status.content = f"Kling error: {exc}"
            status.author = "system"
            await status.update()
    except Exception as exc:
        if status:
            status.content = f"Unexpected Kling error: {exc}"
            status.author = "system"
            await status.update()


async def run_kling_flow(
    *,
    ask_text: TextAskFunc,
    ensure_thread_id: EnsureThreadFunc,
    data_layer,
    add_media_entry: AddMediaFunc,
    call_kling_multi_image_to_video: Callable[..., str],
    call_kling_image_to_video: CallImageToVideoFunc,
    extract_text_input: Callable[[object], str],
    generate_title: Optional[TitleFunc] = None,
    initial_text: str = "",
    initial_files: Optional[List[object]] = None,
) -> None:
    async def ask_choice(prompt: str, options: List[str], default: Optional[str] = None) -> str:
        actions = [cl.Action(name="choice", label=opt, payload={"value": opt}) for opt in options]
        response = await cl.AskActionMessage(content=prompt, actions=actions, timeout=600).send()
        if response is None:
            return default or options[0]
        if isinstance(response, dict):
            payload = response.get("payload")
            if isinstance(payload, dict):
                value = payload.get("value")
                if isinstance(value, str):
                    return value
        if hasattr(response, "get"):
            try:
                payload = response.get("payload")
                if isinstance(payload, dict):
                    value = payload.get("value")
                    if isinstance(value, str):
                        return value
            except Exception:
                pass
        value = getattr(response, "value", None)
        if isinstance(value, str):
            return value
        return default or options[0]

    mode_choice = await ask_choice(
        "Choose Kling generation mode:",
        ["Image → Video", "Multi‑Image → Video"],
        default="Multi‑Image → Video",
    )
    if mode_choice.startswith("Image"):
        await run_kling_image_to_video_flow(
            ask_text=ask_text,
            ensure_thread_id=ensure_thread_id,
            data_layer=data_layer,
            add_media_entry=add_media_entry,
            call_kling_image_to_video=call_kling_image_to_video,
            generate_title=generate_title,
            initial_text=initial_text,
            initial_files=initial_files,
        )
        return

    await run_kling_multi_image_flow(
        ask_text=ask_text,
        ensure_thread_id=ensure_thread_id,
        data_layer=data_layer,
        add_media_entry=add_media_entry,
        call_kling_multi_image_to_video=call_kling_multi_image_to_video,
        extract_text_input=extract_text_input,
        generate_title=generate_title,
        initial_text=initial_text,
        initial_files=initial_files,
    )
