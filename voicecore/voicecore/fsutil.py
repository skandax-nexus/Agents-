"""Crash-safe file writes: temp file, flush, fsync, rename, directory fsync."""
from __future__ import annotations

import os
import uuid
from pathlib import Path


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _tmp_name(path: Path) -> Path:
    """Unique per writer, so concurrent writers of the same file never share a temp file."""
    return path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    tmp = _tmp_name(path)
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _fsync_dir(path.parent)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_torch_save(obj, path: Path) -> None:
    import torch

    path = Path(path)
    tmp = _tmp_name(path)
    try:
        with open(tmp, "wb") as fh:
            torch.save(obj, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _fsync_dir(path.parent)


class EngineLock:
    """Advisory exclusive lock on an engine directory: one process at a time may serve or train it.
    POSIX only (fcntl); elsewhere it is a no-op."""

    def __init__(self, root: Path) -> None:
        self.path = Path(root) / ".lock"
        self._fh = None

    def acquire(self) -> EngineLock:
        try:
            import fcntl
        except ImportError:   # Windows
            return self
        self._fh = open(self.path, "a+")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._fh.close()
            self._fh = None
            raise RuntimeError(f"engine at {self.path.parent} is already in use by another process") from exc
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(str(os.getpid()))
        self._fh.flush()
        return self

    def release(self) -> None:
        if self._fh is not None:
            try:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass
            self._fh.close()
            self._fh = None

    def __enter__(self) -> EngineLock:
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()
