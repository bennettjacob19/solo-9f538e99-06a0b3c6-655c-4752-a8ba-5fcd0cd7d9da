"""Tests for restoring lost jobs (and events) from a completed backup.

Covers: new id assignment, event copying, provenance, idempotent retries
(concurrent / after restart / from a later backup), error messages and full
rollback on write failure.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import date

import pytest

from nybackup import backup, restore, source
from nybackup.web import create_app

PLAN_A = "2026-11-20"
PLAN_B = "2026-11-21"
DAY_A = date(2026, 11, 20)
DAY_B = date(2026, 11, 21)


@pytest.fixture()
def env(tmp_path):
    src = str(tmp_path / "jobs.db")
    out = str(tmp_path / "backups")
    os.makedirs(out, exist_ok=True)
    source.ensure_schema(src)
    source.seed(src, 5)  # jobs 1..5, each with one event
    backup.take_snapshot(src, out, DAY_A)  # backup A
    return {"src": src, "out": out, "tmp": tmp_path}


def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def counts(path: str) -> dict:
    conn = open_db(path)
    try:
        return {
            "jobs": conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0],
            "events": conn.execute("SELECT COUNT(*) FROM job_events").fetchone()[0],
            "restores": conn.execute(
                "SELECT COUNT(*) FROM job_restore_sources"
            ).fetchone()[0],
        }
    finally:
        conn.close()


def test_restore_creates_new_id_and_copies_all_events(env):
    result = restore.restore_job(env["src"], env["out"], PLAN_A, 3)
    assert result.reused is False
    assert result.new_job_id == 6  # next autoincrement id
    assert result.source_key == f"backup:{PLAN_A}/job:3"
    assert result.origin_source_key is None

    conn = open_db(env["src"])
    try:
        origin = open_db(backup.final_path(env["out"], DAY_A)).execute(
            "SELECT * FROM jobs WHERE id=3"
        ).fetchone()
        restored = conn.execute("SELECT * FROM jobs WHERE id=6").fetchone()
        for col in ("title", "site", "status", "priority", "note", "created_utc"):
            assert restored[col] == origin[col]
        # All events follow the job under its new id.
        origin_events = [
            tuple(r)
            for r in open_db(backup.final_path(env["out"], DAY_A)).execute(
                "SELECT ts_utc, message FROM job_events WHERE job_id=3 ORDER BY id"
            )
        ]
        new_events = [
            tuple(r)
            for r in conn.execute(
                "SELECT ts_utc, message FROM job_events WHERE job_id=6 ORDER BY id"
            )
        ]
        assert new_events == origin_events
        assert len(new_events) == result.events

        row = conn.execute(
            "SELECT * FROM job_restore_sources WHERE source_key=?",
            (result.source_key,),
        ).fetchone()
        assert row["new_job_id"] == 6
        assert row["backup_plan_date"] == PLAN_A
        assert row["origin_job_id"] == 3
    finally:
        conn.close()

    # The write counter stays consistent (trigger fired for the restored job).
    stats = source.stats(env["src"])
    assert stats["jobs"] == stats["write_seq"] == 6


def test_restore_is_idempotent_on_repeat_submit(env):
    first = restore.restore_job(env["src"], env["out"], PLAN_A, 3)
    second = restore.restore_job(env["src"], env["out"], PLAN_A, 3)
    assert second.reused is True
    assert second.new_job_id == first.new_job_id == 6
    # Exactly one copy of everything.
    got = counts(env["src"])
    assert got["jobs"] == 6 and got["restores"] == 1


def test_restore_idempotent_after_restart(env):
    first = restore.restore_job(env["src"], env["out"], PLAN_A, 3)
    # A "restart" only reopens the files: the mapping lives in the source DB.
    second = restore.restore_job(env["src"], env["out"], PLAN_A, 3)
    assert second.new_job_id == first.new_job_id
    assert counts(env["src"])["restores"] == 1


def test_concurrent_identical_submits_restore_only_one_copy(env):
    results = []
    errors = []
    barrier = threading.Barrier(6)

    def worker():
        try:
            barrier.wait()
            results.append(restore.restore_job(env["src"], env["out"], PLAN_A, 2))
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(results) == 6
    assert {r.new_job_id for r in results} == {6}
    assert sum(1 for r in results if not r.reused) == 1
    assert counts(env["src"]) == {"jobs": 6, "events": 6, "restores": 1}


def test_later_backup_carries_provenance_and_returns_same_id(env):
    # First restore from backup A: job #3 -> new id 6.
    first = restore.restore_job(env["src"], env["out"], PLAN_A, 3)
    # The next daily snapshot B includes the provenance table.
    backup.take_snapshot(env["src"], env["out"], DAY_B)
    path_b = backup.final_path(env["out"], DAY_B)
    conn_b = open_db(path_b)
    try:
        tables = {
            r[0] for r in conn_b.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "job_restore_sources" in tables
        assert conn_b.execute(
            "SELECT new_job_id FROM job_restore_sources"
        ).fetchone()["new_job_id"] == 6
    finally:
        conn_b.close()

    # Select the already-restored job (id 6 inside B) from the later backup.
    again = restore.restore_job(env["src"], env["out"], PLAN_B, 6)
    assert again.new_job_id == first.new_job_id == 6
    assert again.source_key == first.source_key
    assert again.reused is True
    assert counts(env["src"])["restores"] == 1

    # A different, never-restored job from B gets a fresh id with B's identity.
    other = restore.restore_job(env["src"], env["out"], PLAN_B, 2)
    assert other.reused is False and other.new_job_id == 7
    assert other.source_key == f"backup:{PLAN_B}/job:2"


def test_missing_backup_reports_reason(env):
    with pytest.raises(restore.RestoreError, match="找不到 1999-01-01 的完整备份"):
        restore.restore_job(env["src"], env["out"], "1999-01-01", 1)
    assert counts(env["src"]) == {"jobs": 5, "events": 5, "restores": 0}


def test_corrupt_backup_reports_reason(env):
    corrupt = os.path.join(env["out"], "jobs-2026-11-22.sqlite")
    with open(corrupt, "wb") as fh:
        fh.write(b"this is definitely not a database")
    with pytest.raises(restore.RestoreError, match="已损坏|无法打开|无法读取"):
        restore.restore_job(env["src"], env["out"], "2026-11-22", 1)
    assert counts(env["src"]) == {"jobs": 5, "events": 5, "restores": 0}


def test_unknown_job_in_backup_reports_reason(env):
    with pytest.raises(restore.RestoreError, match="查不到作业 #999"):
        restore.restore_job(env["src"], env["out"], PLAN_A, 999)
    assert counts(env["src"])["restores"] == 0


def test_bad_inputs_are_rejected(env):
    with pytest.raises(restore.RestoreError):
        restore.restore_job(env["src"], env["out"], PLAN_A, 0)
    with pytest.raises(restore.RestoreError):
        restore.restore_job(env["src"], env["out"], PLAN_A, -3)
    with pytest.raises(restore.RestoreError):
        restore.restore_job(env["src"], env["out"], "not-a-date", 1)


def test_write_failure_rolls_back_the_whole_restore(env):
    conn = open_db(env["src"])
    try:
        # Force the provenance insert to fail inside the restore transaction.
        conn.execute(
            "CREATE TRIGGER fail_restore BEFORE INSERT ON job_restore_sources "
            "BEGIN SELECT RAISE(ABORT, 'forced failure'); END"
        )
    finally:
        conn.close()

    before = counts(env["src"])
    with pytest.raises(restore.RestoreError, match="整笔已回滚"):
        restore.restore_job(env["src"], env["out"], PLAN_A, 4)
    after = counts(env["src"])
    assert after == before  # no job, no events, no provenance row left behind


# --------------------------------------------------------------------------- web
@pytest.fixture()
def app(tmp_path):
    source_path = str(tmp_path / "jobs.db")
    source.ensure_schema(source_path)
    source.seed(source_path, 3)
    app = create_app(
        {
            "SOURCE_PATH": source_path,
            "STATE_PATH": str(tmp_path / "state.db"),
            "BACKUP_DIR": str(tmp_path / "backups"),
            "AUTO_START_SCHEDULER": False,
            "TESTING": True,
        }
    )
    backup.take_snapshot(source_path, app.config["BACKUP_DIR"], DAY_A)
    yield app


def test_web_restore_success_and_retry_flash(app):
    client = app.test_client()
    resp = client.post(
        "/restore", data={"plan_date": PLAN_A, "origin_job_id": "2"}
    )
    assert resp.status_code == 302
    body = client.get("/").get_data(as_text=True)
    assert "新编号 #4" in body
    assert f"backup:{PLAN_A}/job:2" in body

    # Retry: same source, same new id, flagged as already restored.
    client.post("/restore", data={"plan_date": PLAN_A, "origin_job_id": "2"})
    body = client.get("/").get_data(as_text=True)
    assert "已补回过，沿用同一新编号 #4" in body
    assert counts(app.config["SOURCE_PATH"])["restores"] == 1


def test_web_restore_error_flash(app):
    client = app.test_client()
    client.post("/restore", data={"plan_date": PLAN_A, "origin_job_id": "999"})
    body = client.get("/").get_data(as_text=True)
    assert "补回失败" in body and "查不到作业 #999" in body


def test_web_restore_validation(app):
    client = app.test_client()
    assert client.post(
        "/restore", data={"plan_date": "bad", "origin_job_id": "1"}
    ).status_code == 302
    assert client.post(
        "/restore", data={"plan_date": PLAN_A, "origin_job_id": "x"}
    ).status_code == 302
    assert counts(app.config["SOURCE_PATH"])["restores"] == 0


def test_api_status_exposes_restores(app):
    client = app.test_client()
    client.post("/restore", data={"plan_date": PLAN_A, "origin_job_id": "1"})
    payload = client.get("/api/status").get_json()
    assert len(payload["restores"]) == 1
    assert payload["restores"][0]["new_job_id"] == 4
    assert payload["restores"][0]["source_key"] == f"backup:{PLAN_A}/job:1"
