


"""Optional PANNs CNN14 embedding wrapper with graceful fallback."""

from __future__ import annotations

import csv
import importlib
import io
import logging
import sys
import types
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("soundscape.panns_embedder")

try:
    import librosa

    LIBROSA_AVAILABLE = True
except Exception:
    librosa = None
    LIBROSA_AVAILABLE = False

try:
    import resampy

    RESAMPY_AVAILABLE = True
except Exception:
    resampy = None
    RESAMPY_AVAILABLE = False

PANNS_MIN_CHECKPOINT_BYTES = 300_000_000


def _resample_audio(waveform: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    if src_sr == tgt_sr:
        return waveform.astype(np.float32)
    if waveform.size == 0:
        return waveform.astype(np.float32)

    if LIBROSA_AVAILABLE:
        try:
            return librosa.resample(waveform.astype(np.float32), orig_sr=src_sr, target_sr=tgt_sr).astype(
                np.float32
            )
        except Exception:
            pass
    if RESAMPY_AVAILABLE:
        try:
            return resampy.resample(waveform.astype(np.float32), src_sr, tgt_sr).astype(np.float32)
        except Exception:
            pass


    x_old = np.linspace(0.0, 1.0, num=waveform.size, endpoint=True)
    new_len = int(round(waveform.size * float(tgt_sr) / float(max(src_sr, 1))))
    new_len = max(1, new_len)
    x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=True)
    return np.interp(x_new, x_old, waveform).astype(np.float32)


def _validate_checkpoint(path: Path) -> Optional[str]:
    if not path.is_file():
        return f"local checkpoint not found: {path.as_posix()}"
    size = int(path.stat().st_size)
    if size <= 0:
        return f"local checkpoint file empty: {path.as_posix()}"

    if size < PANNS_MIN_CHECKPOINT_BYTES:
        return f"local checkpoint too small for Cnn14: {path.as_posix()} ({size} bytes)"
    return None


def _load_local_labels_csv(path: Path) -> Tuple[Optional[List[str]], Optional[List[str]], Optional[str]]:
    if not path.is_file():
        return None, None, f"local labels csv not found: {path.as_posix()}"
    size = int(path.stat().st_size)
    if size <= 0:
        return None, None, f"local labels csv empty: {path.as_posix()}"

    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
    except Exception as exc:
        return None, None, f"local labels csv unreadable: {path.as_posix()} ({exc})"

    if len(rows) <= 1:
        return None, None, f"local labels csv has no data rows: {path.as_posix()}"
    if len(rows[0]) < 3:
        return None, None, f"local labels csv has invalid header: {path.as_posix()}"

    ids: List[str] = []
    labels: List[str] = []
    for row in rows[1:]:
        if len(row) < 3:
            continue
        rid = str(row[1]).strip()
        rlabel = str(row[2]).strip()
        if not rid or not rlabel:
            continue
        ids.append(rid)
        labels.append(rlabel)

    if not labels:
        return None, None, f"local labels csv parsed empty labels: {path.as_posix()}"
    return labels, ids, None


def _build_local_config_module(
    labels: List[str],
    ids: List[str],
    labels_path: Path,
) -> types.ModuleType:
    mod = types.ModuleType("panns_inference.config")
    mod.sample_rate = 32000
    mod.labels_csv_path = labels_path.as_posix()
    mod.labels = labels
    mod.ids = ids
    mod.classes_num = len(labels)
    mod.lb_to_ix = {label: i for i, label in enumerate(labels)}
    mod.ix_to_lb = {i: label for i, label in enumerate(labels)}
    mod.id_to_ix = {rid: i for i, rid in enumerate(ids)}
    mod.ix_to_id = {i: rid for i, rid in enumerate(ids)}
    return mod


def _purge_panns_modules() -> None:
    for name in list(sys.modules.keys()):
        if name == "panns_inference" or name.startswith("panns_inference."):
            sys.modules.pop(name, None)


def _lazy_import_audio_tagging(
    labels: List[str],
    ids: List[str],
    labels_path: Path,
    force_local_resources: bool,
) -> Tuple[Optional[type], Optional[str]]:
    try:
        if force_local_resources:
            _purge_panns_modules()
            sys.modules["panns_inference.config"] = _build_local_config_module(
                labels=labels,
                ids=ids,
                labels_path=labels_path,
            )
        pkg = importlib.import_module("panns_inference")
        audio_tagging_cls = getattr(pkg, "AudioTagging", None)
        if audio_tagging_cls is None:
            return None, "panns import failed: missing AudioTagging class"
        return audio_tagging_cls, None
    except Exception as exc:
        return None, f"panns import failed: {exc}"


class PANNSEmbedder:
    """Optional PANNs embedder that never crashes the whole pipeline."""

    def __init__(
        self,
        export_dims: int = 16,
        enabled: bool = True,
        checkpoint_path: Optional[str] = None,
        labels_path: Optional[str] = None,
        force_local_resources: bool = True,
    ):
        self.export_dims = int(max(1, export_dims))
        self.enabled = bool(enabled)
        self.target_sr = 32000
        self.reason_unavailable = ""
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve()) if checkpoint_path else ""
        self.labels_path = str(Path(labels_path).expanduser().resolve()) if labels_path else ""
        self.force_local_resources = bool(force_local_resources)
        self._tagger = None
        self._device = "cpu"
        self._torch = None

        if not self.enabled:
            self.reason_unavailable = "disabled_by_config"
            return

        checkpoint = Path(self.checkpoint_path) if self.checkpoint_path else None
        labels_csv = Path(self.labels_path) if self.labels_path else None
        if checkpoint is None:
            self.reason_unavailable = "local checkpoint not configured"
            return
        if labels_csv is None:
            self.reason_unavailable = "local labels csv not configured"
            return

        self.checkpoint_path = checkpoint.as_posix()
        self.labels_path = labels_csv.as_posix()

        checkpoint_err = _validate_checkpoint(checkpoint)
        if checkpoint_err:
            self.reason_unavailable = checkpoint_err
            return

        labels, ids, labels_err = _load_local_labels_csv(labels_csv)
        if labels_err:
            self.reason_unavailable = labels_err
            return
        assert labels is not None and ids is not None

        try:
            self._torch = importlib.import_module("torch")
        except Exception as exc:
            self.reason_unavailable = f"panns import failed: {exc}"
            return

        audio_tagging_cls, import_reason = _lazy_import_audio_tagging(
            labels=labels,
            ids=ids,
            labels_path=labels_csv,
            force_local_resources=self.force_local_resources,
        )
        if audio_tagging_cls is None:
            self.reason_unavailable = str(import_reason or "panns import failed: unknown")
            return

        try:
            self._device = "cuda" if self._torch.cuda.is_available() else "cpu"

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self._tagger = audio_tagging_cls(
                    checkpoint_path=self.checkpoint_path,
                    device=self._device,
                )
        except Exception as exc:
            self._tagger = None
            self.reason_unavailable = f"panns runtime init failed: {exc}"
            logger.warning("PANNs 初始化失败，回退占位特征: %s", exc)

    @property
    def is_available(self) -> bool:
        return self._tagger is not None

    def _empty(self, reason: Optional[str] = None) -> Dict[str, object]:
        return {
            "available": False,
            "reason": reason or self.reason_unavailable or "unavailable",
            "embedding_dim": 0,
            "vector": None,
        }

    def extract(self, waveform: np.ndarray, sample_rate: int) -> Dict[str, object]:
        """
        Extract clip-level embedding from waveform.

        Returns dict with:
        - available
        - reason
        - embedding_dim
        - vector (np.ndarray or None)
        """
        if waveform is None or sample_rate is None or sample_rate <= 0 or waveform.size == 0:
            return self._empty("empty_waveform")
        if not self.is_available:
            return self._empty()

        try:
            y = _resample_audio(waveform.astype(np.float32), int(sample_rate), self.target_sr)
            batch = np.expand_dims(y, axis=0)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                _, emb = self._tagger.inference(batch)
            vec = np.asarray(emb[0], dtype=np.float32)
            if vec.size == 0:
                return self._empty("panns runtime inference failed: empty embedding")
            return {
                "available": True,
                "reason": "",
                "embedding_dim": int(vec.shape[0]),
                "vector": vec,
            }
        except Exception as exc:
            logger.warning("PANNs 提取失败，回退占位特征: %s", exc)
            return self._empty(f"panns runtime inference failed: {exc}")

    def to_feature_columns(self, extract_result: Dict[str, object]) -> Dict[str, object]:
        """Convert extract result to stable tabular columns."""
        out: Dict[str, object] = {
            "panns_available": bool(extract_result.get("available", False)),
            "panns_unavailable_reason": str(extract_result.get("reason", "")),
            "panns_embedding_dim": int(extract_result.get("embedding_dim", 0) or 0),
            "panns_emb_mean": float("nan"),
            "panns_emb_std": float("nan"),
            "panns_emb_l2": float("nan"),
        }

        for i in range(self.export_dims):
            out[f"panns_emb_{i:03d}"] = float("nan")

        vec = extract_result.get("vector")
        if not isinstance(vec, np.ndarray) or vec.size == 0:
            return out

        out["panns_emb_mean"] = float(np.mean(vec))
        out["panns_emb_std"] = float(np.std(vec))
        out["panns_emb_l2"] = float(np.linalg.norm(vec))

        dims = min(self.export_dims, vec.size)
        for i in range(dims):
            out[f"panns_emb_{i:03d}"] = float(vec[i])
        return out


def run_local_panns_self_check(
    checkpoint_path: str,
    labels_path: str,
    force_local_resources: bool = True,
    export_dims: int = 16,
) -> Dict[str, Any]:
    """
    Validate project-local PANNs resource wiring without running full pipeline.
    """
    embedder = PANNSEmbedder(
        export_dims=export_dims,
        enabled=True,
        checkpoint_path=checkpoint_path,
        labels_path=labels_path,
        force_local_resources=force_local_resources,
    )
    if not embedder.is_available:
        return {
            "ok": False,
            "reason": embedder.reason_unavailable,
            "checkpoint_path": embedder.checkpoint_path,
            "labels_path": embedder.labels_path,
            "resource_mode": "local_only" if force_local_resources else "default",
            "summary": f"[check_panns] status=fail reason={embedder.reason_unavailable}",
        }


    dummy = np.zeros(32000, dtype=np.float32)
    infer = embedder.extract(dummy, 32000)
    if not bool(infer.get("available", False)):
        reason = str(infer.get("reason", "panns runtime inference failed: unknown"))
        return {
            "ok": False,
            "reason": reason,
            "checkpoint_path": embedder.checkpoint_path,
            "labels_path": embedder.labels_path,
            "resource_mode": "local_only" if force_local_resources else "default",
            "summary": f"[check_panns] status=fail reason={reason}",
        }

    return {
        "ok": True,
        "reason": "",
        "checkpoint_path": embedder.checkpoint_path,
        "labels_path": embedder.labels_path,
        "resource_mode": "local_only" if force_local_resources else "default",
        "embedding_dim": int(infer.get("embedding_dim", 0) or 0),
        "summary": (
            "[check_panns] status=pass "
            f"checkpoint={embedder.checkpoint_path} labels={embedder.labels_path}"
        ),
    }
