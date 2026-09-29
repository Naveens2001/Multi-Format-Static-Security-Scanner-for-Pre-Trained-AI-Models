"""
model_scanner.presentation.web.executor
=======================================

Bounded asynchronous scan executor and in-memory scan job registry (Phase 9).

Enforces:
- A ceiling on actively executing scans (default 2).
- A ceiling on waiting scans (HTTP 429 when capacity is exhausted).
- Bounded result store (max 50 jobs with 1-hour TTL eviction).
- Discrete lifecycle states: QUEUED, ACQUIRING, SCANNING, COMPLETED, ERROR,
  CANCELLED.
- Automatic cleanup of temporary upload directories.

Queue accounting
----------------
Waiting depth is DERIVED from job state rather than tracked in a counter.
A counter that is only decremented inside the worker body leaks permanently
whenever every worker slot is held by a long-running scan: the queued jobs
never start, the counter never drops, and the server refuses all new work
until the process is restarted. Deriving the number cannot drift, because
every terminal path sets a status.

Cancellation
------------
A job that has not started yet is cancelled outright and its slot is freed
immediately. A job already executing cannot be killed - Python offers no safe
way to stop a running thread - so it is marked CANCELLED, its result is
discarded when it eventually returns, and its temporary files are removed.
``cancelled_while_running`` records which of the two happened so callers can
tell the user the truth.

Watchdogs
---------
- A job waiting longer than ``max_queue_wait_seconds`` is failed rather than
  waiting forever.
- A job executing longer than ``max_scan_seconds`` is reported as timed out so
  the UI stops waiting. The underlying thread is left to unwind on its own.
"""

from __future__ import annotations

import concurrent.futures
import enum
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from model_scanner.core.models import ScanResult
from model_scanner.presentation.json_export import JSONReportExporter

# Defaults. Override per-instance, or via ScannerConfig at the call site.
DEFAULT_MAX_ACTIVE: int = 4
DEFAULT_MAX_QUEUED: int = 10
DEFAULT_MAX_STORED_RESULTS: int = 50
DEFAULT_TTL_SECONDS: int = 3600
DEFAULT_MAX_SCAN_SECONDS: int = 1800
DEFAULT_MAX_QUEUE_WAIT_SECONDS: int = 900


class ScanJobStatus(enum.StrEnum):
    """Discrete lifecycle states for a static security scan job."""

    QUEUED = "QUEUED"
    ACQUIRING = "ACQUIRING"
    SCANNING = "SCANNING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


#: States in which a job is finished and occupies no capacity.
TERMINAL_STATUSES: frozenset[ScanJobStatus] = frozenset(
    {ScanJobStatus.COMPLETED, ScanJobStatus.ERROR, ScanJobStatus.CANCELLED}
)

#: States in which a job is waiting for a worker but is not yet running.
WAITING_STATUSES: frozenset[ScanJobStatus] = frozenset(
    {ScanJobStatus.QUEUED, ScanJobStatus.ACQUIRING}
)


class ScanQueueFullError(Exception):
    """Raised when active and queued scan capacity is exhausted."""

    def __init__(
        self,
        message: str = "Scan queue is full.",
        waiting: int = 0,
        running: int = 0,
        max_queued: int = DEFAULT_MAX_QUEUED,
        max_active: int = DEFAULT_MAX_ACTIVE,
    ) -> None:
        super().__init__(message)
        self.error_code = "SCAN_QUEUE_FULL"
        self.waiting = waiting
        self.running = running
        self.max_queued = max_queued
        self.max_active = max_active


class ScanNotCancellableError(Exception):
    """Raised when a cancel is requested for a job that has already finished."""

    def __init__(self, message: str = "Scan has already finished.") -> None:
        super().__init__(message)
        self.error_code = "SCAN_NOT_CANCELLABLE"


@dataclass
class ScanJob:
    """Represents an asynchronous scan job and its immutable result."""

    scan_id: str
    target_name: str
    status: ScanJobStatus
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    result: dict[str, object] | None = None
    scan_result_obj: ScanResult | None = None  # Raw object for HTML/PDF rendering
    error_message: str | None = None
    error_code: str | None = None
    cleanup_paths: list[Path] = field(default_factory=list)
    future: concurrent.futures.Future[None] | None = field(default=None, repr=False)
    cancel_requested: bool = False
    cancelled_while_running: bool = False

    @property
    def is_terminal(self) -> bool:
        """True when the job has finished and holds no capacity."""
        return self.status in TERMINAL_STATUSES

    @property
    def is_waiting(self) -> bool:
        """True when the job is waiting for a worker but has not started."""
        return self.status in WAITING_STATUSES

    def to_dict(self) -> dict[str, object]:
        """Convert job state to JSON-serializable dictionary."""
        return {
            "scan_id": self.scan_id,
            "target_name": self.target_name,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "cancelled_while_running": self.cancelled_while_running,
            "result": self.result,
        }

    def to_summary(self) -> dict[str, object]:
        """Compact representation for queue listings (omits the report body)."""
        return {
            "scan_id": self.scan_id,
            "target_name": self.target_name,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "cancellable": not self.is_terminal,
        }


class BoundedScanExecutor:
    """Bounded scan execution manager enforcing strict active and queue ceilings."""

    def __init__(
        self,
        max_active: int = DEFAULT_MAX_ACTIVE,
        max_queued: int = DEFAULT_MAX_QUEUED,
        max_stored_results: int = DEFAULT_MAX_STORED_RESULTS,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_scan_seconds: int = DEFAULT_MAX_SCAN_SECONDS,
        max_queue_wait_seconds: int = DEFAULT_MAX_QUEUE_WAIT_SECONDS,
    ) -> None:
        self._max_active = max(1, int(max_active))
        self._max_queued = max(1, int(max_queued))
        self._max_stored_results = max(1, int(max_stored_results))
        self._ttl_seconds = int(ttl_seconds)
        self._max_scan_seconds = int(max_scan_seconds)
        self._max_queue_wait_seconds = int(max_queue_wait_seconds)

        self._thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_active,
            thread_name_prefix="ModelScannerWorker",
        )
        self._jobs: dict[str, ScanJob] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # capacity
    # ------------------------------------------------------------------

    def _waiting_locked(self) -> int:
        """Count jobs waiting for a worker (must hold _lock)."""
        return sum(1 for j in self._jobs.values() if j.is_waiting)

    def _running_locked(self) -> int:
        """Count jobs currently executing (must hold _lock)."""
        return sum(1 for j in self._jobs.values() if j.status is ScanJobStatus.SCANNING)

    def capacity(self) -> dict[str, object]:
        """Return a snapshot of queue occupancy and limits."""
        with self._lock:
            self._reap_locked()
            waiting = self._waiting_locked()
            running = self._running_locked()
            return {
                "waiting": waiting,
                "running": running,
                "max_queued": self._max_queued,
                "max_active": self._max_active,
                "accepting": waiting < self._max_queued,
                "stored_jobs": len(self._jobs),
            }

    # ------------------------------------------------------------------
    # submission
    # ------------------------------------------------------------------

    def submit_scan(
        self,
        target_name: str,
        scan_fn: Callable[[], ScanResult],
        cleanup_paths: list[Path] | None = None,
        initial_status: ScanJobStatus = ScanJobStatus.QUEUED,
        preset_scan_id: str | None = None,
    ) -> str:
        """Submit a scan function for asynchronous execution.

        Parameters
        ----------
        preset_scan_id:
            Optional caller-supplied UUID to use as the job scan_id. When
            provided the caller must also pass this value into the engine
            (e.g. via ScanOrchestrator.scan(..., scan_id=...)) so that the
            job ID and ScanResult.scan_id are identical.

        Returns
        -------
        str
            Unique scan_id (UUID4).

        Raises
        ------
        ScanQueueFullError
            If the number of waiting scans has reached ``max_queued``.
        """
        scan_id = preset_scan_id if preset_scan_id is not None else str(uuid.uuid4())
        paths_to_clean = list(cleanup_paths or [])

        with self._lock:
            self._reap_locked()
            self._cleanup_expired_jobs_locked()

            waiting = self._waiting_locked()
            if waiting >= self._max_queued:
                for path in paths_to_clean:
                    self._remove_path(path)
                raise ScanQueueFullError(
                    f"Scan queue capacity reached ({waiting} of {self._max_queued} "
                    f"waiting, {self._running_locked()} running). Cancel a waiting "
                    f"scan or wait for a running scan to finish.",
                    waiting=waiting,
                    running=self._running_locked(),
                    max_queued=self._max_queued,
                    max_active=self._max_active,
                )

            job = ScanJob(
                scan_id=scan_id,
                target_name=target_name,
                status=initial_status,
                cleanup_paths=paths_to_clean,
            )
            self._jobs[scan_id] = job

        try:
            future = self._thread_pool.submit(
                self._worker_routine, scan_id, scan_fn, paths_to_clean
            )
        except RuntimeError as exc:
            # Pool already shut down - do not leave a phantom waiting job behind.
            with self._lock:
                err_job: ScanJob | None = self._jobs.get(scan_id)
                if err_job:
                    err_job.status = ScanJobStatus.ERROR
                    err_job.error_code = "EXECUTOR_UNAVAILABLE"
                    err_job.error_message = str(exc)
                    err_job.completed_at = time.time()
            for path in paths_to_clean:
                self._remove_path(path)
            raise

        with self._lock:
            job2: ScanJob | None = self._jobs.get(scan_id)
            if job2:
                job2.future = future

        return scan_id

    def _worker_routine(
        self,
        scan_id: str,
        scan_fn: Callable[[], ScanResult],
        cleanup_paths: list[Path],
    ) -> None:
        """Execute a scan inside a pool worker."""
        with self._lock:
            job = self._jobs.get(scan_id)
            if job is None:
                return
            if job.cancel_requested:
                # Cancelled after submit but before the worker picked it up.
                job.status = ScanJobStatus.CANCELLED
                job.completed_at = time.time()
                self._cleanup_paths_now(cleanup_paths)
                return
            job.status = ScanJobStatus.SCANNING
            job.started_at = time.time()

        try:
            result = scan_fn()
            serialized: dict[str, object] = JSONReportExporter.to_dict(result)

            with self._lock:
                job = self._jobs.get(scan_id)
                if job is None:
                    return
                if job.cancel_requested:
                    # Discard the report - the caller stopped waiting for it.
                    job.status = ScanJobStatus.CANCELLED
                    job.completed_at = time.time()
                else:
                    job.result = serialized
                    job.scan_result_obj = result
                    job.status = ScanJobStatus.COMPLETED
                    job.completed_at = time.time()
        except Exception as exc:
            error_code = getattr(exc, "error_code", "SCAN_EXECUTION_ERROR")
            with self._lock:
                job = self._jobs.get(scan_id)
                if job is None:
                    return
                if job.cancel_requested:
                    job.status = ScanJobStatus.CANCELLED
                else:
                    job.status = ScanJobStatus.ERROR
                    job.error_code = str(error_code)
                    job.error_message = str(exc)
                job.completed_at = time.time()
        finally:
            self._cleanup_paths_now(cleanup_paths)

    # ------------------------------------------------------------------
    # cancellation
    # ------------------------------------------------------------------

    def cancel_scan(self, scan_id: str) -> dict[str, object]:
        """Cancel a queued or running scan.

        A job that has not started is cancelled outright and frees its slot
        immediately. A job already executing is marked cancelled; its result is
        discarded when the worker returns, but the worker slot is not freed
        until then, because a running Python thread cannot be safely killed.

        Returns
        -------
        dict
            ``{"scan_id", "status", "was_running", "freed_slot"}``.

        Raises
        ------
        KeyError
            If ``scan_id`` is unknown.
        ScanNotCancellableError
            If the job has already reached a terminal state.
        """
        with self._lock:
            job = self._jobs.get(scan_id)
            if job is None:
                raise KeyError(scan_id)
            if job.is_terminal:
                raise ScanNotCancellableError(
                    f"Scan is already {job.status.value}; there is nothing to cancel."
                )

            job.cancel_requested = True
            was_running = job.status is ScanJobStatus.SCANNING
            future = job.future
            cleanup = list(job.cleanup_paths)

        if was_running:
            # The worker is mid-scan. It will observe cancel_requested when it
            # returns, discard the report and clean up. Report honestly that
            # the slot is not free yet.
            with self._lock:
                job = self._jobs.get(scan_id)
                if job is not None:
                    job.cancelled_while_running = True
                    status = job.status.value
                else:
                    status = ScanJobStatus.CANCELLED.value
            return {
                "scan_id": scan_id,
                "status": status,
                "was_running": True,
                "freed_slot": False,
            }

        # Not started yet. Pull it out of the pool if we still can; either way
        # the job is terminal from here, so its slot is released immediately.
        if future is not None:
            future.cancel()

        with self._lock:
            job = self._jobs.get(scan_id)
            if job is None:
                raise KeyError(scan_id)
            job.status = ScanJobStatus.CANCELLED
            job.completed_at = time.time()
            status = job.status.value

        self._cleanup_paths_now(cleanup)

        return {
            "scan_id": scan_id,
            "status": status,
            "was_running": False,
            "freed_slot": True,
        }

    def cancel_all_waiting(self) -> dict[str, object]:
        """Cancel every job that has not started executing.

        This is the escape hatch when all worker slots are held by long-running
        scans and the queue has filled behind them.
        """
        with self._lock:
            waiting_ids = [j.scan_id for j in self._jobs.values() if j.is_waiting]

        cancelled: list[str] = []
        for scan_id in waiting_ids:
            try:
                outcome = self.cancel_scan(scan_id)
            except (KeyError, ScanNotCancellableError):
                continue
            if outcome["status"] == ScanJobStatus.CANCELLED.value:
                cancelled.append(scan_id)

        cap = self.capacity()
        return {"cancelled": cancelled, "cancelled_count": len(cancelled), "capacity": cap}

    # ------------------------------------------------------------------
    # lookup and maintenance
    # ------------------------------------------------------------------

    def get_job(self, scan_id: str) -> ScanJob | None:
        """Look up a scan job by scan_id, applying watchdogs first."""
        with self._lock:
            self._reap_locked()
            return self._jobs.get(scan_id)

    def list_jobs(self) -> list[ScanJob]:
        """Return all active and completed scan jobs."""
        with self._lock:
            self._reap_locked()
            return list(self._jobs.values())

    def _reap_locked(self) -> None:
        """Fail jobs that have waited or run past their limits (must hold _lock)."""
        now = time.time()
        for job in self._jobs.values():
            if job.is_terminal:
                continue

            if job.is_waiting and self._max_queue_wait_seconds > 0:
                if now - job.created_at > self._max_queue_wait_seconds:
                    if job.future is not None:
                        job.future.cancel()
                    job.status = ScanJobStatus.ERROR
                    job.error_code = "SCAN_QUEUE_TIMEOUT"
                    job.error_message = (
                        f"Scan waited more than {self._max_queue_wait_seconds}s for a "
                        f"free worker and was abandoned."
                    )
                    job.completed_at = now
                    self._cleanup_paths_now(job.cleanup_paths)
                    continue

            if (
                job.status is ScanJobStatus.SCANNING
                and self._max_scan_seconds > 0
                and job.started_at is not None
                and now - job.started_at > self._max_scan_seconds
            ):
                job.cancel_requested = True
                job.cancelled_while_running = True
                job.status = ScanJobStatus.ERROR
                job.error_code = "SCAN_TIMEOUT"
                job.error_message = (
                    f"Scan exceeded the {self._max_scan_seconds}s limit and was "
                    f"abandoned. The worker is still unwinding in the background."
                )
                job.completed_at = now

    def _cleanup_expired_jobs_locked(self) -> None:
        """Evict jobs older than TTL or exceeding max stored results (must hold _lock)."""
        now = time.time()
        expired = [
            sid
            for sid, job in self._jobs.items()
            if job.is_terminal and (now - job.created_at) > self._ttl_seconds
        ]
        for sid in expired:
            self._jobs.pop(sid, None)

        if len(self._jobs) > self._max_stored_results:
            finished = sorted(
                (j for j in self._jobs.values() if j.is_terminal),
                key=lambda j: j.created_at,
            )
            overflow = len(self._jobs) - self._max_stored_results
            for job_to_evict in finished[:overflow]:
                self._jobs.pop(job_to_evict.scan_id, None)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @classmethod
    def _cleanup_paths_now(cls, paths: list[Path]) -> None:
        """Remove every path in ``paths``, ignoring failures."""
        for path in paths:
            cls._remove_path(path)

    @staticmethod
    def _remove_path(path: Path) -> None:
        """Safely delete a file or directory tree."""
        try:
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the background executor and clean up resources."""
        self._thread_pool.shutdown(wait=wait, cancel_futures=True)


__all__ = [
    "TERMINAL_STATUSES",
    "WAITING_STATUSES",
    "BoundedScanExecutor",
    "ScanJob",
    "ScanJobStatus",
    "ScanNotCancellableError",
    "ScanQueueFullError",
]
