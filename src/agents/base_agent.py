


"""Base class for strict-JSON agent calls with local cache and fallback."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

logger = logging.getLogger("agents.base_agent")


class BaseAgent:
    """Reusable strict-JSON LLM agent wrapper."""

    agent_name: str = "base_agent"
    output_schema: Mapping[str, Any] = {"type": "object", "required": []}
    system_prompt: str = "你是结构化推理助手。"

    def __init__(
        self,
        client: Any,
        model_name: str,
        cache_dir: str,
        max_retries: int = 2,
        temperature: float = 0.1,
    ) -> None:
        self.client = client
        self.model_name = str(model_name)
        self.max_retries = int(max_retries)
        self.temperature = float(temperature)
        self.cache_dir = Path(cache_dir) / self.agent_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def build_payload(self, context_payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Build model input payload from orchestrator context."""
        return dict(context_payload)

    def fallback_output(self, context_payload: Mapping[str, Any], reason: str) -> Dict[str, Any]:
        """Deterministic fallback output when LLM is unavailable/failed."""
        sid = int(context_payload.get("segment_id", -1))
        return {
            "segment_id": sid,
            "fallback": True,
            "fallback_reason": str(reason),
        }

    def _required_keys(self) -> Sequence[str]:
        req = self.output_schema.get("required", [])
        if isinstance(req, list):
            return [str(x) for x in req]
        return []

    def _input_hash(self, payload: Mapping[str, Any]) -> str:
        text = json.dumps(
            {
                "agent_name": self.agent_name,
                "model_name": self.model_name,
                "output_schema": dict(self.output_schema),
                "payload": dict(payload),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cache_path(self, segment_id: int) -> Path:
        sid = max(-1, int(segment_id))
        if sid >= 0:
            return self.cache_dir / f"segment_{sid:06d}.json"
        return self.cache_dir / "segment_unknown.json"

    def _load_cache(self, cache_path: Path, input_hash: str) -> Dict[str, Any]:
        if not cache_path.is_file():
            return {"hit": False}
        try:
            obj = json.loads(cache_path.read_text(encoding="utf-8"))
            if not isinstance(obj, dict):
                return {"hit": False}
            if str(obj.get("input_hash", "")) != str(input_hash):
                return {"hit": False}
            if str(obj.get("model_name", "")) != self.model_name:
                return {"hit": False}
            output = obj.get("output")
            if not isinstance(output, dict):
                return {"hit": False}
            return {"hit": True, "output": output, "raw": obj}
        except Exception as exc:
            logger.warning("cache read failed agent=%s path=%s err=%s", self.agent_name, cache_path.as_posix(), exc)
            return {"hit": False}

    def _write_cache(self, cache_path: Path, input_hash: str, output: Mapping[str, Any], attempts: int) -> None:
        payload = {
            "agent_name": self.agent_name,
            "model_name": self.model_name,
            "input_hash": input_hash,
            "attempts": int(attempts),
            "output": dict(output),
            "cached_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _normalize_output(self, output: Mapping[str, Any], fallback: Mapping[str, Any]) -> Dict[str, Any]:
        out = dict(output)
        fb = dict(fallback)
        for key in self._required_keys():
            if key not in out:
                out[key] = fb.get(key)
        return out

    def run(self, context_payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Run one agent call with cache and graceful fallback."""
        sid = int(context_payload.get("segment_id", -1))
        model_payload = self.build_payload(context_payload)
        input_hash = self._input_hash(model_payload)
        cache_path = self._cache_path(sid)

        cached = self._load_cache(cache_path, input_hash=input_hash)
        if bool(cached.get("hit", False)):
            return {
                "agent": self.agent_name,
                "segment_id": sid,
                "status": "ok",
                "cache_hit": True,
                "attempts": int(cached.get("raw", {}).get("attempts", 0)),
                "error": "",
                "model_name": self.model_name,
                "output": dict(cached.get("output", {})),
            }

        if (self.client is None) or (not bool(getattr(self.client, "available", False))):
            reason = str(getattr(self.client, "unavailable_reason", "client_unavailable"))
            fallback = self.fallback_output(context_payload, reason=f"api_unavailable:{reason}")
            fallback = self._normalize_output(fallback, fallback)
            return {
                "agent": self.agent_name,
                "segment_id": sid,
                "status": "fallback",
                "cache_hit": False,
                "attempts": 0,
                "error": f"api_unavailable:{reason}",
                "model_name": self.model_name,
                "output": fallback,
            }

        response = self.client.request_json(
            model=self.model_name,
            system_prompt=str(self.system_prompt),
            user_payload=model_payload,
            output_schema=self.output_schema,
            max_retries=self.max_retries,
            temperature=self.temperature,
        )
        if bool(response.get("ok", False)) and isinstance(response.get("json"), dict):
            raw_output = dict(response.get("json", {}))
            fallback = self.fallback_output(context_payload, reason="")
            normalized = self._normalize_output(raw_output, fallback=fallback)

            try:
                self._write_cache(
                    cache_path=cache_path,
                    input_hash=input_hash,
                    output=normalized,
                    attempts=int(response.get("attempts", 0)),
                )
            except Exception as exc:
                logger.warning(
                    "cache write failed agent=%s path=%s err=%s",
                    self.agent_name,
                    cache_path.as_posix(),
                    exc,
                )
            return {
                "agent": self.agent_name,
                "segment_id": sid,
                "status": "ok",
                "cache_hit": False,
                "attempts": int(response.get("attempts", 0)),
                "error": "",
                "model_name": self.model_name,
                "output": normalized,
            }

        reason = str(response.get("error", "model_request_failed"))
        fallback = self.fallback_output(context_payload, reason=f"request_failed:{reason}")
        fallback = self._normalize_output(fallback, fallback=fallback)
        return {
            "agent": self.agent_name,
            "segment_id": sid,
            "status": "fallback",
            "cache_hit": False,
            "attempts": int(response.get("attempts", 0)),
            "error": f"request_failed:{reason}",
            "model_name": self.model_name,
            "output": fallback,
        }

