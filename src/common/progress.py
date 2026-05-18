


"""Unified terminal progress helpers for the main pipeline and post stages."""

from __future__ import annotations

import logging
import sys
from typing import Optional

logger = logging.getLogger("common.progress")

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    _RICH_AVAILABLE = True
except Exception:
    Console = None
    Progress = None
    SpinnerColumn = None
    TextColumn = None
    BarColumn = None
    TaskProgressColumn = None
    TimeElapsedColumn = None
    _RICH_AVAILABLE = False


class ProgressTaskHandle:
    """Thin task wrapper with graceful no-op behavior."""

    def __init__(
        self,
        manager: "UnifiedProgressManager",
        task_id: Optional[int],
        description: str,
        total: Optional[float] = None,
    ) -> None:
        self.manager = manager
        self.task_id = task_id
        self.description = description
        self.total = float(total) if total is not None else None

    def update(
        self,
        *,
        completed: Optional[float] = None,
        advance: Optional[float] = None,
        total: Optional[float] = None,
        description: Optional[str] = None,
    ) -> None:
        if total is not None:
            self.total = float(total)
        if description:
            self.description = description
        self.manager.update_task(
            self.task_id,
            completed=completed,
            advance=advance,
            total=total,
            description=description or self.description,
        )

    def update_percent(self, percent: float, description: Optional[str] = None) -> None:
        pct = max(0.0, min(100.0, float(percent)))
        self.update(total=100.0, completed=pct, description=description)

    def advance(self, step: float = 1.0, description: Optional[str] = None) -> None:
        self.update(advance=float(step), description=description)

    def finish(self, status: str = "done", description: Optional[str] = None) -> None:
        label = description or self.description
        prefix = {
            "done": "done",
            "skipped": "skipped",
            "failed": "failed",
        }.get(status, status)
        final_desc = f"{label} [{prefix}]"
        if self.total is not None:
            self.update(total=self.total, completed=self.total, description=final_desc)
        else:
            self.update(description=final_desc)
        self.manager.log_summary(final_desc, status=status)


class UnifiedProgressManager:
    """Shared progress manager for terminal-friendly rich progress bars."""

    def __init__(self, *, enabled: Optional[bool] = None, transient: bool = False) -> None:
        if enabled is None:
            enabled = bool(_RICH_AVAILABLE and hasattr(sys.stderr, "isatty") and sys.stderr.isatty())
        self.enabled = bool(enabled and _RICH_AVAILABLE)
        self.transient = bool(transient)
        self.console = Console(stderr=True) if self.enabled and Console is not None else None
        self._progress = None

    def __enter__(self) -> "UnifiedProgressManager":
        if self.enabled and Progress is not None:
            self._progress = Progress(
                SpinnerColumn(style="cyan"),
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(bar_width=32, complete_style="green"),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                transient=self.transient,
                console=self.console,
            )
            self._progress.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None

    def add_task(self, description: str, total: Optional[float] = 100.0) -> ProgressTaskHandle:
        task_id: Optional[int] = None
        if self._progress is not None:
            task_id = self._progress.add_task(description, total=total)
        return ProgressTaskHandle(self, task_id=task_id, description=description, total=total)

    def update_task(
        self,
        task_id: Optional[int],
        *,
        completed: Optional[float] = None,
        advance: Optional[float] = None,
        total: Optional[float] = None,
        description: Optional[str] = None,
    ) -> None:
        if self._progress is None or task_id is None:
            return
        kwargs = {}
        if completed is not None:
            kwargs["completed"] = float(completed)
        if advance is not None:
            kwargs["advance"] = float(advance)
        if total is not None:
            kwargs["total"] = float(total)
        if description is not None:
            kwargs["description"] = str(description)
        if kwargs:
            self._progress.update(task_id, **kwargs)

    def log_summary(self, message: str, *, status: str = "done") -> None:
        if self.console is None:
            logger.info(message)


def create_progress_manager(*, enabled: Optional[bool] = None, transient: bool = False) -> UnifiedProgressManager:
    """Factory kept small so callers do not import rich directly."""
    return UnifiedProgressManager(enabled=enabled, transient=transient)
