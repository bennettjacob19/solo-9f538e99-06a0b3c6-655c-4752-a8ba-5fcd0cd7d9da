"""Restore a lost job (and all of its events) from a daily backup snapshot.

Flow
----
An operator picks a *completed* backup on the page and enters the original job
number.  The lost job is copied back into the live source database, where it
receives a **new** autoincremented id; every one of its events is copied under
that new id.

Provenance and idempotence
--------------------------
The source database keeps a ``job_restore_sources`` row for every restored job.
Its unique ``source_key`` identifies the lost job inside its backup:

* if the job inside the backup already carries a restore source, that recorded
  source is reused (so a job restored a second time from a later backup lands
  on the same new id);
* otherwise the source is identified by the backup's plan date and the
  job's original id, e.g. ``backup:2026-11-20/job:7``.

Because provenance lives in the source database it is part of every later
snapshot, and the same source submitted twice -- concurrently, retried after a
restart, or selected again from a subsequent backup -- always returns the same
new id and inserts exactly one copy.

Everything for one restore runs inside a single ``BEGIN IMMEDIATE``
transaction; any failure rolls the whole operation back and leaves the source
database untouched.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from . import backup
from .db import connect, transaction

# Columns copied verbatim from a backup's ``jobs`` row (id is reassigned).
JOB_COPY_COLUMNS = ("title", "site", "status", "priority", "note", "created_utc")
EVENT_COPY_COLUMNS = ("ts_utc", "message")

_restore_lock = threading.Lock()


class RestoreError(RuntimeError):
    """A user-facing restore failure (missing/corrupt backup, unknown job, ...)."""


@dataclass(frozen=True)
class RestoreResult:
    new_job_id: int
    source_key: str
    backup_plan_date: str
    origin_job_id: int
    origin_source_key: Optional[str]
    origin_note: str
    events: int
    reused: bool  # True when an existing restore mapping was returned as-is


def make_source_key(plan_date: str, origin_job_id: int) -> str:
    """The default source identity for a job without recorded provenance."""
    return f"backup:{plan_date}/job:{int(origin_job_id)}"


def _open_backup(path: str) -> sqlite3.Connection:
    """Open a backup read-only and make sure it is a usable jobs database."""
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        raise RestoreError("备份文件缺失或为空，无法补回")
    try:
        conn = connect(path, readonly=True)
    except sqlite3.DatabaseError as exc:
        raise RestoreError(f"备份文件无法打开（已损坏或不是 SQLite 数据库）：{exc}") from exc
    try:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RestoreError(f"备份文件已损坏，完整性检查未通过：{integrity}")
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        missing = {"jobs", "job_events"} - tables
        if missing:
            raise RestoreError(
                "备份文件缺少必需的数据表：" + "、".join(sorted(missing))
            )
    except sqlite3.DatabaseError as exc:
        conn.close()
        raise RestoreError(f"备份文件已损坏，无法读取：{exc}") from exc
    return conn


def _read_origin_provenance(
    backup_conn: sqlite3.Connection, origin_job_id: int
) -> Optional[sqlite3.Row]:
    """Return the backup's own recorded restore source for the job, if any."""
    tables = {
        row[0]
        for row in backup_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    if "job_restore_sources" not in tables:
        return None
    return backup_conn.execute(
        "SELECT source_key, origin_note FROM job_restore_sources WHERE new_job_id=?",
        (origin_job_id,),
    ).fetchone()


def resolve_source(
    backup_conn: sqlite3.Connection, plan_date: str, origin_job_id: int
) -> tuple[str, Optional[str], str]:
    """Resolve ``(source_key, origin_source_key, origin_note)`` for a backup job.

    A recorded source (from an earlier restore already captured inside this
    backup) is reused; otherwise the backup plan date plus the original job id
    become the identity.
    """
    recorded = _read_origin_provenance(backup_conn, origin_job_id)
    if recorded is not None:
        note = f"沿用备份内已记录的补回来源 {recorded['source_key']}"
        return recorded["source_key"], recorded["source_key"], note
    return make_source_key(plan_date, origin_job_id), None, ""


def restore_job(
    source_path: str,
    backup_dir: str,
    plan_date: str,
    origin_job_id: int,
) -> RestoreResult:
    """Restore one job and all of its events into the live source database.

    Idempotent: the same provenance key always maps to the same new job id.
    Raises :class:`RestoreError` with a Chinese explanation for every expected
    failure; the source database is left unchanged on any error.
    """
    try:
        int(origin_job_id)
    except (TypeError, ValueError):
        raise RestoreError("作业编号必须是整数")
    origin_job_id = int(origin_job_id)
    if origin_job_id <= 0:
        raise RestoreError("作业编号必须为正整数")

    try:
        datetime.strptime(plan_date, "%Y-%m-%d")
    except (TypeError, ValueError):
        raise RestoreError("备份计划日期格式应为 YYYY-MM-DD")

    path = backup.list_backup_files(backup_dir).get(plan_date)
    if path is None:
        raise RestoreError(f"找不到 {plan_date} 的完整备份文件，无法补回")

    # Serialise restores against each other; the DB unique constraint plus the
    # in-transaction lookup below still guarantee correctness regardless.
    with _restore_lock:
        backup_conn = _open_backup(path)
        try:
            origin = backup_conn.execute(
                "SELECT id, title, site, status, priority, note, created_utc "
                "FROM jobs WHERE id=?",
                (origin_job_id,),
            ).fetchone()
            if origin is None:
                raise RestoreError(
                    f"备份 {plan_date} 中查不到作业 #{origin_job_id}"
                )
            event_rows = list(
                backup_conn.execute(
                    "SELECT ts_utc, message FROM job_events WHERE job_id=? ORDER BY id",
                    (origin_job_id,),
                )
            )
            source_key, origin_source_key, origin_note = resolve_source(
                backup_conn, plan_date, origin_job_id
            )
        finally:
            backup_conn.close()

        # Make sure the live database has the provenance table (old source DBs).
        from .source import ensure_schema

        ensure_schema(source_path)
        src_conn = connect(source_path)
        try:
            with transaction(src_conn):
                existing = src_conn.execute(
                    "SELECT * FROM job_restore_sources WHERE source_key=?",
                    (source_key,),
                ).fetchone()
                if existing is not None:
                    # Same source again: do not insert a second copy; report the
                    # same new id every time.
                    return RestoreResult(
                        new_job_id=existing["new_job_id"],
                        source_key=source_key,
                        backup_plan_date=plan_date,
                        origin_job_id=origin_job_id,
                        origin_source_key=existing["origin_source_key"],
                        origin_note=existing["origin_note"],
                        events=src_conn.execute(
                            "SELECT COUNT(*) FROM job_events WHERE job_id=?",
                            (existing["new_job_id"],),
                        ).fetchone()[0],
                        reused=True,
                    )

                now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
                cur = src_conn.execute(
                    "INSERT INTO jobs(title, site, status, priority, note, created_utc) "
                    "VALUES (?,?,?,?,?,?)",
                    tuple(origin[col] for col in JOB_COPY_COLUMNS),
                )
                new_job_id = cur.lastrowid
                for ev in event_rows:
                    src_conn.execute(
                        "INSERT INTO job_events(job_id, ts_utc, message) VALUES (?,?,?)",
                        (new_job_id, ev["ts_utc"], ev["message"]),
                    )
                src_conn.execute(
                    "INSERT INTO job_restore_sources(source_key, backup_plan_date, "
                    "origin_job_id, origin_source_key, origin_note, new_job_id, restored_utc) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (
                        source_key,
                        plan_date,
                        origin_job_id,
                        origin_source_key,
                        origin_note,
                        new_job_id,
                        now_iso,
                    ),
                )
        except RestoreError:
            raise
        except sqlite3.IntegrityError as exc:
            # Lost a race with a concurrent identical submit: the other
            # transaction has already claimed this source.  Read its mapping so
            # the caller still gets the same new id.
            row = src_conn.execute(
                "SELECT * FROM job_restore_sources WHERE source_key=?",
                (source_key,),
            ).fetchone()
            if row is not None:
                return RestoreResult(
                    new_job_id=row["new_job_id"],
                    source_key=source_key,
                    backup_plan_date=plan_date,
                    origin_job_id=origin_job_id,
                    origin_source_key=row["origin_source_key"],
                    origin_note=row["origin_note"],
                    events=src_conn.execute(
                        "SELECT COUNT(*) FROM job_events WHERE job_id=?",
                        (row["new_job_id"],),
                    ).fetchone()[0],
                    reused=True,
                )
            raise RestoreError(f"写入失败，整笔已回滚：{exc}") from exc
        except sqlite3.DatabaseError as exc:
            raise RestoreError(f"写入源库失败，整笔已回滚：{exc}") from exc
        finally:
            src_conn.close()

        return RestoreResult(
            new_job_id=new_job_id,
            source_key=source_key,
            backup_plan_date=plan_date,
            origin_job_id=origin_job_id,
            origin_source_key=origin_source_key,
            origin_note=origin_note,
            events=len(event_rows),
            reused=False,
        )


def list_restores(source_path: str, limit: int = 25) -> list[sqlite3.Row]:
    """Recent restore records joined with the new job's title for the page."""
    if not os.path.exists(source_path):
        return []
    conn = connect(source_path, readonly=True)
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "job_restore_sources" not in tables:
            return []
        return list(
            conn.execute(
                "SELECT r.*, j.title AS new_job_title "
                "FROM job_restore_sources r "
                "LEFT JOIN jobs j ON j.id = r.new_job_id "
                "ORDER BY r.id DESC LIMIT ?",
                (limit,),
            )
        )
    finally:
        conn.close()
