


"""Zhipu API client wrapper for strict-JSON agent reasoning."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger("agents.zhipu_client")


def _load_local_env_file(root_dir: Optional[Path] = None) -> Dict[str, str]:
    """Load KEY=VALUE pairs from project-local apikey.env."""
    env_map: Dict[str, str] = {}
    base = root_dir or Path(__file__).resolve().parents[2]
    env_path = base / "apikey.env"
    if not env_path.is_file():
        return env_map
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value:
                env_map[key] = value
    except Exception as exc:
        logger.warning("load apikey.env failed: %s", exc)
    return env_map


def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _content_to_text(choice_content: Any) -> str:
    if isinstance(choice_content, str):
        return choice_content.strip()
    if isinstance(choice_content, dict):
        return json.dumps(choice_content, ensure_ascii=False)
    if isinstance(choice_content, list):
        parts = []
        for item in choice_content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p).strip()
    return str(choice_content).strip()


class ZhipuAgentClient:
    """Thin wrapper around zhipuai SDK with retries and strict JSON parsing."""

    def __init__(
        self,
        client: Any = None,
        available: bool = False,
        unavailable_reason: str = "",
    ) -> None:
        self._client = client
        self.available = bool(available)
        self.unavailable_reason = str(unavailable_reason or "")

    @classmethod
    def from_apikey_env(cls, root_dir: Optional[Path] = None) -> "ZhipuAgentClient":
        local_env = _load_local_env_file(root_dir=root_dir)
        api_key = (
            os.getenv("ZHIPUAI_API_KEY")
            or os.getenv("ZHIPU_API_KEY")
            or local_env.get("ZHIPUAI_API_KEY")
            or local_env.get("ZHIPU_API_KEY")
        )
        if not api_key:
            return cls(available=False, unavailable_reason="missing_api_key:ZHIPUAI_API_KEY/ZHIPU_API_KEY")
        try:
            from zhipuai import ZhipuAI

            return cls(client=ZhipuAI(api_key=api_key), available=True, unavailable_reason="")
        except Exception as exc:
            return cls(available=False, unavailable_reason=f"zhipu_import_or_init_failed:{exc}")

    def _call_once(
        self,
        model: str,
        system_prompt: str,
        user_payload: Mapping[str, Any],
        output_schema: Mapping[str, Any],
        temperature: float,
    ) -> Dict[str, Any]:
        assert self._client is not None
        user_text = (
            "你是结构化推理代理。请严格输出一个JSON对象，不要输出Markdown、解释或额外文本。\n"
            "输出必须满足 schema（键名一致）：\n"
            f"{json.dumps(dict(output_schema), ensure_ascii=False)}\n"
            "输入数据如下（JSON）：\n"
            f"{json.dumps(dict(user_payload), ensure_ascii=False)}"
        )
        messages = [
            {"role": "system", "content": str(system_prompt)},
            {"role": "user", "content": user_text},
        ]

        kwargs = {
            "model": str(model),
            "temperature": float(temperature),
            "messages": messages,
        }


        last_exc = None
        for with_response_format in (True, False):
            try:
                call_kwargs = dict(kwargs)
                if with_response_format:
                    call_kwargs["response_format"] = {"type": "json_object"}
                resp = self._client.chat.completions.create(**call_kwargs)
                choice = resp.choices[0].message.content
                text = _content_to_text(choice)
                obj = _extract_json_obj(text)
                if obj is None:
                    raise ValueError("model_output_not_json_object")
                return {"ok": True, "json": obj, "raw_text": text}
            except Exception as exc:
                last_exc = exc
                continue
        return {"ok": False, "error": f"zhipu_request_failed:{last_exc}"}

    def request_json(
        self,
        model: str,
        system_prompt: str,
        user_payload: Mapping[str, Any],
        output_schema: Mapping[str, Any],
        max_retries: int = 2,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """Request strict JSON response with retry and exponential backoff."""
        if not self.available or self._client is None:
            return {
                "ok": False,
                "json": None,
                "error": self.unavailable_reason or "client_unavailable",
                "attempts": 0,
            }

        attempts = 0
        max_attempts = max(1, int(max_retries) + 1)
        last_error = ""
        for attempt in range(1, max_attempts + 1):
            attempts = attempt
            result = self._call_once(
                model=model,
                system_prompt=system_prompt,
                user_payload=user_payload,
                output_schema=output_schema,
                temperature=float(temperature),
            )
            if bool(result.get("ok", False)):
                return {
                    "ok": True,
                    "json": result.get("json"),
                    "raw_text": result.get("raw_text", ""),
                    "error": "",
                    "attempts": attempts,
                }
            last_error = str(result.get("error", "unknown_error"))
            if attempt < max_attempts:
                sleep_s = min(4.0, 0.8 * (2 ** (attempt - 1)))
                time.sleep(sleep_s)

        return {
            "ok": False,
            "json": None,
            "raw_text": "",
            "error": last_error or "unknown_error",
            "attempts": attempts,
        }

