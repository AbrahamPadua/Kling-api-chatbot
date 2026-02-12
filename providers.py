import os
import time
from typing import List, Dict, Optional, Any
import requests
from dotenv import load_dotenv
import jwt

load_dotenv()

class ProviderError(Exception):
    """Raised when a provider call fails."""


def _get_key(env_name: str) -> str:
    key = os.getenv(env_name)
    if not key:
        raise ProviderError(f"Missing API key: set {env_name} in .env")
    return key


def _get_kling_token() -> str:
    access_key = _get_key("KLING_ACCESS_KEY")
    secret_key = _get_key("KLING_SECRET_KEY")
    headers = {
        "alg": "HS256",
        "typ": "JWT",
    }
    payload = {
        "iss": access_key,
        "exp": int(time.time()) + 1800,
        "nbf": int(time.time()) - 5,
    }
    try:
        token = jwt.encode(payload, secret_key, headers=headers)
    except Exception as exc:
        raise ProviderError(f"Failed to generate Kling JWT token: {exc}") from exc
    if isinstance(token, bytes):
        return token.decode("utf-8")
    return token


def call_provider(
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    system_prompt: str = "",
    max_tokens: int = 1024,
) -> str:
    provider = provider.lower()
    if provider == "claude":
        return _call_anthropic(model, messages, temperature, system_prompt, max_tokens)
    if provider == "gpt":
        return _call_openai(model, messages, temperature, system_prompt, max_tokens)
    if provider == "gemini":
        return _call_gemini(model, messages, temperature, system_prompt, max_tokens)
    if provider == "kling":
        raise ProviderError(
            "Kling OmniVideo requires structured inputs (images/videos). "
            "Use the /kling flow in the app instead of the chat provider call."
        )
    raise ProviderError(f"Unsupported provider: {provider}")


def list_anthropic_models(limit: int = 100, max_pages: int = 5) -> List[str]:
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise ProviderError("Missing API key: set ANTHROPIC_API_KEY (or CLAUDE_API_KEY) in .env")
    base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    anthropic_version = os.getenv("ANTHROPIC_VERSION", "2023-06-01")
    headers = {
        "x-api-key": api_key,
        "anthropic-version": anthropic_version,
        "content-type": "application/json",
    }

    model_ids: List[str] = []
    after_id: Optional[str] = None

    try:
        for _ in range(max_pages):
            params: Dict[str, Any] = {"limit": limit}
            if after_id:
                params["after_id"] = after_id
            resp = requests.get(f"{base_url}/v1/models", headers=headers, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("data", [])
            if not isinstance(items, list):
                break
            for item in items:
                if not isinstance(item, dict):
                    continue
                model_id = item.get("id")
                if isinstance(model_id, str) and model_id.startswith("claude"):
                    model_ids.append(model_id)

            if not data.get("has_more"):
                break
            last_id = data.get("last_id")
            if not isinstance(last_id, str) or not last_id:
                break
            after_id = last_id
    except requests.HTTPError as exc:
        detail = ""
        request_id = None
        status_code = None
        try:
            status_code = resp.status_code
            detail = resp.text
            request_id = resp.headers.get("request-id")
        except Exception:
            pass
        request_id_part = f" request-id={request_id}" if request_id else ""
        raise ProviderError(
            f"Anthropic model list error {status_code}: {detail or exc}.{request_id_part}"
        ) from exc
    except requests.RequestException as exc:
        raise ProviderError(f"Anthropic model list network error: {exc}") from exc

    # de-duplicate while preserving order
    deduped = list(dict.fromkeys(model_ids))
    if not deduped:
        raise ProviderError("Anthropic model list returned no Claude models.")
    return deduped


def call_kling_omni_video(
    prompt: str,
    image_list: Optional[List[Dict[str, str]]] = None,
    video_list: Optional[List[Dict[str, str]]] = None,
    model_name: str = "kling-video-o1",
    mode: str = "pro",
    aspect_ratio: str = "1:1",
    duration: str = "7",
    poll_interval_s: int = 5,
    timeout_s: int = 600,
) -> str:
    api_key = _get_kling_token()
    url = "https://api-singapore.klingai.com/v1/videos/omni-video"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model_name": model_name,
        "prompt": prompt,
        "mode": mode,
        "aspect_ratio": aspect_ratio,
        "duration": duration,
    }
    if image_list:
        payload["image_list"] = image_list
    if video_list:
        payload["video_list"] = video_list

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise ProviderError(f"Kling create error: {data}")
        task_id = data.get("data", {}).get("task_id")
        if not task_id:
            raise ProviderError(f"Kling create missing task_id: {data}")
    except requests.RequestException as exc:
        raise ProviderError(f"Kling create request failed: {exc}") from exc

    start = time.time()
    while True:
        if time.time() - start > timeout_s:
            raise ProviderError("Kling job timed out while polling.")
        try:
            poll_resp = requests.get(url, headers=headers, params={"task_id": task_id}, timeout=60)
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()
            if poll_data.get("code") != 0:
                raise ProviderError(f"Kling poll error: {poll_data}")
            status = poll_data.get("data", {}).get("task_status")
            if status == "succeed":
                videos = poll_data.get("data", {}).get("task_result", {}).get("videos", [])
                if videos:
                    url_val = videos[0].get("url")
                    if url_val:
                        return url_val
                raise ProviderError(f"Kling succeeded but no video URL found: {poll_data}")
            if status in {"failed", "error"}:
                raise ProviderError(f"Kling job failed: {poll_data}")
        except requests.RequestException as exc:
            raise ProviderError(f"Kling poll request failed: {exc}") from exc
        time.sleep(poll_interval_s)


def call_kling_multi_image_to_video(
    prompt: str,
    image_list: List[Dict[str, str]],
    negative_prompt: Optional[str] = None,
    model_name: str = "kling-v1-6",
    mode: str = "std",
    aspect_ratio: str = "16:9",
    duration: str = "5",
    callback_url: Optional[str] = None,
    external_task_id: Optional[str] = None,
    poll_interval_s: int = 5,
    timeout_s: int = 600,
) -> str:
    if not image_list:
        raise ProviderError("Kling multi-image requires at least one image in image_list.")

    api_key = _get_kling_token()
    base_url = "https://api-singapore.klingai.com"
    create_url = f"{base_url}/v1/videos/multi-image2video"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model_name": model_name,
        "prompt": prompt,
        "image_list": image_list,
        "mode": mode,
        "duration": duration,
        "aspect_ratio": aspect_ratio,
    }
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if callback_url:
        payload["callback_url"] = callback_url
    if external_task_id:
        payload["external_task_id"] = external_task_id

    try:
        resp = requests.post(create_url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise ProviderError(f"Kling create error: {data}")
        task_id = data.get("data", {}).get("task_id")
        if not task_id:
            raise ProviderError(f"Kling create missing task_id: {data}")
    except requests.RequestException as exc:
        raise ProviderError(f"Kling create request failed: {exc}") from exc

    start = time.time()
    while True:
        if time.time() - start > timeout_s:
            raise ProviderError("Kling job timed out while polling.")
        try:
            poll_url = f"{base_url}/v1/videos/multi-image2video/{task_id}"
            poll_resp = requests.get(poll_url, headers=headers, timeout=60)
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()
            if poll_data.get("code") != 0:
                raise ProviderError(f"Kling poll error: {poll_data}")
            status = poll_data.get("data", {}).get("task_status")
            if status == "succeed":
                videos = poll_data.get("data", {}).get("task_result", {}).get("videos", [])
                if videos:
                    url_val = videos[0].get("url")
                    if url_val:
                        return url_val
                raise ProviderError(f"Kling succeeded but no video URL found: {poll_data}")
            if status in {"failed", "error"}:
                message = poll_data.get("data", {}).get("task_status_msg")
                raise ProviderError(f"Kling job failed: {message or poll_data}")
        except requests.RequestException as exc:
            raise ProviderError(f"Kling poll request failed: {exc}") from exc
        time.sleep(poll_interval_s)


def call_kling_image_to_video(
    image: str,
    prompt: Optional[str] = None,
    image_tail: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    model_name: str = "kling-v1-6",
    mode: str = "std",
    duration: str = "5",
    cfg_scale: Optional[float] = None,
    callback_url: Optional[str] = None,
    external_task_id: Optional[str] = None,
    poll_interval_s: int = 5,
    timeout_s: int = 600,
) -> str:
    if not image and not image_tail:
        raise ProviderError("Kling image-to-video requires image or image_tail.")

    api_key = _get_kling_token()
    base_url = "https://api-singapore.klingai.com"
    create_url = f"{base_url}/v1/videos/image2video"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model_name": model_name,
        "mode": mode,
        "duration": duration,
    }
    if image:
        payload["image"] = image
    if image_tail:
        payload["image_tail"] = image_tail
    if prompt:
        payload["prompt"] = prompt
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if cfg_scale is not None:
        payload["cfg_scale"] = cfg_scale
    if callback_url:
        payload["callback_url"] = callback_url
    if external_task_id:
        payload["external_task_id"] = external_task_id

    try:
        resp = requests.post(create_url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise ProviderError(f"Kling create error: {data}")
        task_id = data.get("data", {}).get("task_id")
        if not task_id:
            raise ProviderError(f"Kling create missing task_id: {data}")
    except requests.RequestException as exc:
        raise ProviderError(f"Kling create request failed: {exc}") from exc

    start = time.time()
    while True:
        if time.time() - start > timeout_s:
            raise ProviderError("Kling job timed out while polling.")
        try:
            poll_url = f"{base_url}/v1/videos/image2video/{task_id}"
            poll_resp = requests.get(poll_url, headers=headers, timeout=60)
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()
            if poll_data.get("code") != 0:
                raise ProviderError(f"Kling poll error: {poll_data}")
            status = poll_data.get("data", {}).get("task_status")
            if status == "succeed":
                videos = poll_data.get("data", {}).get("task_result", {}).get("videos", [])
                if videos:
                    url_val = videos[0].get("url")
                    if url_val:
                        return url_val
                raise ProviderError(f"Kling succeeded but no video URL found: {poll_data}")
            if status in {"failed", "error"}:
                message = poll_data.get("data", {}).get("task_status_msg")
                raise ProviderError(f"Kling job failed: {message or poll_data}")
        except requests.RequestException as exc:
            raise ProviderError(f"Kling poll request failed: {exc}") from exc
        time.sleep(poll_interval_s)


def get_kling_task_result(task_id: str) -> Dict[str, Any]:
    if not task_id:
        raise ProviderError("Kling task_id is required.")
    api_key = _get_kling_token()
    base_url = "https://api-singapore.klingai.com"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    endpoints = [
        f"{base_url}/v1/videos/image2video/{task_id}",
        f"{base_url}/v1/videos/multi-image2video/{task_id}",
    ]
    last_error: Optional[str] = None
    for url in endpoints:
        try:
            resp = requests.get(url, headers=headers, timeout=60)
            if resp.status_code == 404:
                last_error = f"Task not found at {url}"
                continue
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                last_error = f"Kling task error: {data}"
                continue
            status = data.get("data", {}).get("task_status")
            videos = data.get("data", {}).get("task_result", {}).get("videos", [])
            video_url = None
            if isinstance(videos, list) and videos:
                video_url = videos[0].get("url")
            return {
                "status": status,
                "video_url": video_url,
                "raw": data,
            }
        except requests.RequestException as exc:
            last_error = f"Kling task request failed: {exc}"
            continue
    raise ProviderError(last_error or "Kling task lookup failed for the provided task_id.")


def _call_anthropic(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    system_prompt: str,
    max_tokens: int,
) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise ProviderError("Missing API key: set ANTHROPIC_API_KEY (or CLAUDE_API_KEY) in .env")
    base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    url = f"{base_url}/v1/messages"
    anthropic_version = os.getenv("ANTHROPIC_VERSION", "2023-06-01")
    headers = {
        "x-api-key": api_key,
        "anthropic-version": anthropic_version,
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            for msg in messages
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system_prompt:
        payload["system"] = system_prompt
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("content", [])
        if content and isinstance(content, list):
            # Anthropic returns list of content blocks
            texts = [c.get("text", "") for c in content if isinstance(c, dict)]
            return "\n".join(t for t in texts if t)
        return str(data)
    except requests.HTTPError as exc:
        detail = ""
        request_id = None
        status_code = None
        try:
            status_code = resp.status_code
            detail = resp.text
            request_id = resp.headers.get("request-id")
        except Exception:
            pass
        if status_code == 404 and "not_found_error" in detail and "model:" in detail:
            fallback_model = os.getenv("ANTHROPIC_FALLBACK_MODEL", "claude-opus-4-6")
            if fallback_model and fallback_model != model:
                return _call_anthropic(
                    fallback_model,
                    messages,
                    temperature,
                    system_prompt,
                    max_tokens,
                )
        request_id_part = f" request-id={request_id}" if request_id else ""
        raise ProviderError(
            f"Anthropic error {status_code}: {detail or exc}.{request_id_part}"
        ) from exc
    except requests.RequestException as exc:
        raise ProviderError(f"Anthropic network error: {exc}") from exc


def _call_openai(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    system_prompt: str,
    max_tokens: int,
) -> str:
    api_key = _get_key("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    oai_messages: List[Dict[str, str]] = []
    if system_prompt:
        oai_messages.append({"role": "system", "content": system_prompt})
    oai_messages.extend(
        {"role": msg.get("role", "user"), "content": msg.get("content", "")}
        for msg in messages
    )
    payload = {
        "model": model,
        "messages": oai_messages,
        "temperature": temperature,
        # Newer GPT models prefer max_completion_tokens; avoid max_tokens to prevent 400s
        "max_completion_tokens": max_tokens,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        choice = data.get("choices", [{}])[0] if isinstance(data.get("choices"), list) else {}
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        content = message.get("content")
        finish_reason = choice.get("finish_reason")

        if content:
            return content

        # No content returned; surface a descriptive error instead of empty text
        if finish_reason == "length":
            raise ProviderError(
                "OpenAI returned no content and hit the max token limit (finish_reason=length). "
                "Try a smaller max token setting or a different model. Response: "
                f"{data}"
            )

        if "error" in data:
            raise ProviderError(f"OpenAI error payload: {data['error']}")

        text_body = None
        try:
            text_body = resp.text
        except Exception:
            text_body = None
        raise ProviderError(
            "OpenAI returned no content. Full response: "
            f"{data or text_body or '(empty response)'}"
        )
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = resp.text
        except Exception:
            detail = ""
        if resp.status_code == 404:
            raise ProviderError(
                f"OpenAI 404: check model name '{model}' or base URL ({url}). Response: {detail}"
            ) from exc
        raise ProviderError(f"OpenAI error {resp.status_code}: {detail or exc}") from exc
    except requests.RequestException as exc:
        raise ProviderError(f"OpenAI network error: {exc}") from exc


def _call_gemini(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    system_prompt: str,
    max_tokens: int,
) -> str:
    api_key = _get_key("GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    def to_part(msg: Dict[str, str]):
        return {"text": msg.get("content", "")}

    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        role = "user" if role == "user" else "model"
        contents.append({"role": role, "parts": [to_part(msg)]})

    payload: Dict[str, object] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if system_prompt:
        payload["system_instruction"] = {"parts": [{"text": system_prompt}]}
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
            return "\n".join(t for t in texts if t)
        return str(data)
    except requests.RequestException as exc:
        raise ProviderError(f"Gemini error: {exc}") from exc
