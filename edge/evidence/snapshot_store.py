"""Atomic local storage for rendered alert snapshots."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from edge.evidence.clip_recorder import CLIP_STORE_DIR_ENV, DEFAULT_CLIP_STORE_DIR


@dataclass(frozen=True, slots=True)
class StoredSnapshot:
    snapshot_id: str
    path: str
    sha256: str
    size_bytes: int
    mime_type: str
    captured_at: str
    camera_id: str
    edge_event_id: str | None


@dataclass(frozen=True, slots=True)
class SnapshotConflictError(Exception):
    snapshot_id: str

    def __str__(self) -> str:
        return f"snapshot {self.snapshot_id} conflicts with existing bytes or metadata"


@dataclass(frozen=True, slots=True)
class SnapshotStore:
    store_dir: Path = Path(os.environ.get(CLIP_STORE_DIR_ENV, DEFAULT_CLIP_STORE_DIR))

    def store(
        self,
        jpeg: bytes,
        *,
        snapshot_id: str,
        captured_at: str,
        camera_id: str,
        edge_event_id: str | None,
    ) -> StoredSnapshot:
        """Atomically compare-or-insert one immutable snapshot identity."""
        relative_path = self._relative_path(camera_id, captured_at, snapshot_id)
        destination = self.store_dir / relative_path
        identity_key = hashlib.sha256(snapshot_id.encode("utf-8")).hexdigest()
        lock_name = f"{identity_key[:2]}.lock"
        lock_directory = self._open_directory(Path(".snapshot-locks"), create=True)
        try:
            lock_descriptor = self._open_file(
                lock_directory, lock_name, os.O_RDWR | os.O_CREAT, 0o600
            )
            with os.fdopen(lock_descriptor, "a+b") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                return self._store_locked(
                    jpeg,
                    snapshot_id=snapshot_id,
                    captured_at=captured_at,
                    camera_id=camera_id,
                    edge_event_id=edge_event_id,
                    relative_path=relative_path,
                    destination=destination,
                    identity_key=identity_key,
                )
        finally:
            os.close(lock_directory)

    def _store_locked(
        self,
        jpeg: bytes,
        *,
        snapshot_id: str,
        captured_at: str,
        camera_id: str,
        edge_event_id: str | None,
        relative_path: Path,
        destination: Path,
        identity_key: str,
    ) -> StoredSnapshot:
        expected = StoredSnapshot(
            snapshot_id=snapshot_id,
            path=relative_path.as_posix(),
            sha256=hashlib.sha256(jpeg).hexdigest(),
            size_bytes=len(jpeg),
            mime_type="image/jpeg",
            captured_at=captured_at,
            camera_id=camera_id,
            edge_event_id=edge_event_id,
        )
        metadata_relative = Path(".snapshot-identities") / f"{identity_key}.json"
        metadata_path = self.store_dir / metadata_relative
        metadata_content = self._read_file(metadata_relative)
        if metadata_content is not None:
            try:
                actual = StoredSnapshot(**json.loads(metadata_content.decode("utf-8")))
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise SnapshotConflictError(snapshot_id) from exc
            if actual != expected:
                raise SnapshotConflictError(snapshot_id)
        else:
            # The identity binding is authoritative before any path-dependent blob
            # becomes visible. A retry after a blob failure must resume this binding,
            # never rebind the same snapshot_id to different immutable metadata.
            self._write_atomic(
                metadata_path,
                json.dumps(asdict(expected), sort_keys=True, separators=(",", ":")).encode("utf-8"),
            )

        existing = self._read_file(relative_path)
        if existing is not None:
            if existing != jpeg:
                raise SnapshotConflictError(snapshot_id)
            # A prior post-replace durability barrier may have failed. Retrying an
            # idempotent store must complete that barrier before reporting success.
            directory = self._open_directory(relative_path.parent, create=False)
            try:
                self._fsync_directory(directory)
            finally:
                os.close(directory)
        else:
            self._write_atomic(destination, jpeg)

        return expected

    def _write_atomic(self, destination: Path, content: bytes) -> None:
        relative = self._relative_to_root(destination)
        directory = self._open_directory(relative.parent, create=True)
        temporary = f".{relative.name}.{uuid4().hex}.tmp"
        primary_error: BaseException | None = None
        try:
            descriptor = self._open_file(
                directory, temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            with os.fdopen(descriptor, "wb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            try:
                self._reject_symlink(directory, relative.name)
            except FileNotFoundError:
                pass
            os.replace(
                temporary,
                relative.name,
                src_dir_fd=directory,
                dst_dir_fd=directory,
            )
            self._fsync_directory(directory)
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(f"temporary snapshot cleanup failed: {cleanup_error}")
            os.close(directory)

    def _read_file(self, relative: Path) -> bytes | None:
        directory = self._open_directory(relative.parent, create=True)
        try:
            try:
                self._reject_symlink(directory, relative.name)
            except FileNotFoundError:
                return None
            descriptor = self._open_file(directory, relative.name, os.O_RDONLY)
            with os.fdopen(descriptor, "rb") as input_file:
                return input_file.read()
        finally:
            os.close(directory)

    def _open_directory(self, relative: Path, *, create: bool) -> int:
        descriptor = self._open_store_root()
        descriptors = [descriptor]
        try:
            for component in relative.parts:
                _validate_path_component(component)
                try:
                    self._reject_symlink(descriptor, component)
                except FileNotFoundError:
                    if not create:
                        raise
                    try:
                        os.mkdir(component, mode=0o700, dir_fd=descriptor)
                    except FileExistsError:
                        # Another identity may create a shared parent concurrently.
                        # The descriptor-relative O_NOFOLLOW open below revalidates it.
                        pass
                child = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
                descriptors.append(child)
                descriptor = child
        except BaseException:
            os.close(descriptor)
            raise
        else:
            try:
                for opened in reversed(descriptors):
                    self._fsync_directory(opened)
            except BaseException:
                os.close(descriptor)
                raise
            return descriptor
        finally:
            for opened in descriptors[:-1]:
                os.close(opened)

    def _open_store_root(self) -> int:
        try:
            root_stat = os.lstat(self.store_dir)
        except FileNotFoundError:
            try:
                self.store_dir.mkdir(mode=0o700)
            except FileExistsError:
                # A concurrent writer may have created the root after lstat.
                pass
            root_stat = os.lstat(self.store_dir)
        if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
            raise ValueError(f"snapshot store root is not a directory: {self.store_dir}")
        return os.open(self.store_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)

    @staticmethod
    def _open_file(directory: int, name: str, flags: int, mode: int = 0o600) -> int:
        return os.open(name, flags | os.O_NOFOLLOW, mode, dir_fd=directory)

    @staticmethod
    def _reject_symlink(directory: int, name: str) -> None:
        entry = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if stat.S_ISLNK(entry.st_mode):
            raise ValueError(f"snapshot path component is a symlink: {name}")
        if not stat.S_ISREG(entry.st_mode) and not stat.S_ISDIR(entry.st_mode):
            raise ValueError(f"snapshot path component has unsupported type: {name}")

    def _relative_to_root(self, path: Path) -> Path:
        try:
            relative = path.relative_to(self.store_dir)
        except ValueError as exc:
            raise ValueError(f"snapshot path escapes store root: {path}") from exc
        if not relative.parts:
            raise ValueError("snapshot destination cannot be the store root")
        return relative

    @staticmethod
    def _fsync_directory(descriptor: int) -> None:
        os.fsync(descriptor)

    @staticmethod
    def _relative_path(camera_id: str, captured_at: str, snapshot_id: str) -> Path:
        date = datetime.fromisoformat(captured_at.replace("Z", "+00:00")).date().isoformat()
        camera_key = hashlib.sha256(camera_id.encode("utf-8")).hexdigest()[:16]
        snapshot_key = hashlib.sha256(snapshot_id.encode("utf-8")).hexdigest()
        return Path("snapshots") / camera_key / date / f"{snapshot_key}.jpg"


def _validate_path_component(component: str) -> None:
    if component in ("", ".", ".."):
        raise ValueError(f"invalid snapshot path component: {component!r}")
