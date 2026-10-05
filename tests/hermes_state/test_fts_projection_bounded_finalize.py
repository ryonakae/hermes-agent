"""Bounded, restartable FTS projection finalization contracts."""

import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_common import (
    FTS_PROJECTION_PENDING_KEY,
    FTS_TRIGRAM_PROJECTION_PENDING_KEY,
    _FTS_TRIGGERS,
)
from hermes_state_fts import _FTS_TRIGRAM_TRIGGERS


_CHUNK_KEYS = ("fts_projection_high_water", "fts_projection_progress")


def _trigger_names(db, names):
    placeholders = ",".join("?" for _ in names)
    return {
        row[0]
        for row in db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' AND name IN ("
            f"{placeholders})",
            tuple(names),
        )
    }


def _install_old_projection_surfaces(db):
    conn = db._conn
    for name in _FTS_TRIGGERS:
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
    conn.execute("DROP TABLE IF EXISTS messages_fts")
    conn.execute("DROP VIEW IF EXISTS messages_fts_src")
    conn.execute(
        "CREATE VIEW messages_fts_src AS SELECT id, content, tool_name, tool_calls FROM messages"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE messages_fts USING fts5("
        "content, tool_name, tool_calls, content='messages_fts_src', content_rowid='id')"
    )
    conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
    if db._trigram_available:
        for name in _FTS_TRIGRAM_TRIGGERS:
            conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        conn.execute("DROP TABLE IF EXISTS messages_fts_trigram")
        conn.execute("DROP VIEW IF EXISTS messages_fts_trigram_src")
        conn.execute(
            "CREATE VIEW messages_fts_trigram_src AS "
            "SELECT id, role, content, tool_name FROM messages WHERE role <> 'tool'"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE messages_fts_trigram USING fts5("
            "content, tool_name, content='messages_fts_trigram_src', "
            "content_rowid='id', tokenize='trigram')"
        )
        conn.execute("INSERT INTO messages_fts_trigram(messages_fts_trigram) VALUES('rebuild')")
    conn.commit()


def _prepare_pending_projection(path):
    db = SessionDB(db_path=path)
    if not db._fts_enabled or not db._trigram_available:
        db.close()
        pytest.skip("SQLite FTS5 and trigram tokenizer required")
    db.create_session("s", source="cli")
    rows = []
    for index in range(4):
        rows.append(
            db.append_message(
                "s",
                role="user",
                content=[
                    {"type": "text", "text": f"bounded marker {index}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,RAW{index}"}},
                ],
            )
        )
    _install_old_projection_surfaces(db)
    db._conn.execute("UPDATE messages SET fts_content = NULL")
    db._conn.commit()
    db._mark_projection_rebuilds_for_backfill()
    with db._lock:
        db._migrate_misaligned_fts_source(db._conn, legacy=False)
        db._migrate_trigram_projection_source(db._conn)
        db._conn.commit()
    db._backfill_projection_content()
    assert db.get_meta(FTS_PROJECTION_PENDING_KEY) == "1"
    assert db.get_meta(FTS_TRIGRAM_PROJECTION_PENDING_KEY) == "1"
    assert _trigger_names(db, _FTS_TRIGGERS) == set()
    return db, rows


def _strict_integrity_probe(db):
    db._conn.execute(
        "INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)"
    )


def test_projection_finalization_is_bounded_and_restartable_after_interruption(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    db, rows = _prepare_pending_projection(path)
    db._FTS_REBUILD_CHUNK_ROWS = 1
    trace = []
    db._conn.set_trace_callback(trace.append)
    original_execute_write = db._execute_write
    calls = {"count": 0}

    def interrupt_after_one_chunk(fn, patience_s=None):
        calls["count"] += 1
        if calls["count"] > 2:
            raise RuntimeError("simulated bounded projection interruption")
        return original_execute_write(fn, patience_s=patience_s)

    monkeypatch.setattr(db, "_execute_write", interrupt_after_one_chunk)
    with pytest.raises(RuntimeError, match="bounded projection interruption"):
        db._finalize_projection_surfaces()
    db._conn.set_trace_callback(None)
    assert calls["count"] > 2
    assert db.get_meta(FTS_PROJECTION_PENDING_KEY) == "1"
    assert db.get_meta("fts_projection_progress") is not None
    assert _trigger_names(db, _FTS_TRIGGERS) == set()
    assert not any("messages_fts(messages_fts) VALUES('rebuild')" in sql for sql in trace)
    db.close()

    reopened = SessionDB(db_path=path)
    try:
        assert reopened.get_meta(FTS_PROJECTION_PENDING_KEY) == "1"
        assert reopened.optimize_fts_storage(vacuum=False)["ok"] is True
        assert reopened.get_meta(FTS_PROJECTION_PENDING_KEY) is None
        assert reopened.get_meta(FTS_TRIGRAM_PROJECTION_PENDING_KEY) is None
        assert _trigger_names(reopened, _FTS_TRIGGERS) == set(_FTS_TRIGGERS)
        assert [row["id"] for row in reopened.search_messages("bounded marker 3")] == [rows[-1]]
        _strict_integrity_probe(reopened)
    finally:
        reopened.close()


def test_projection_dirty_journal_preserves_concurrent_update_and_delete(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    db, rows = _prepare_pending_projection(path)
    db._FTS_REBUILD_CHUNK_ROWS = 1
    original_execute_write = db._execute_write
    calls = {"count": 0}

    def interrupt_after_first_chunk(fn, patience_s=None):
        calls["count"] += 1
        if calls["count"] > 2:
            raise RuntimeError("stop after first durable projection chunk")
        return original_execute_write(fn, patience_s=patience_s)

    monkeypatch.setattr(db, "_execute_write", interrupt_after_first_chunk)
    with pytest.raises(RuntimeError, match="first durable projection chunk"):
        db._finalize_projection_surfaces()
    db.close()

    writer = sqlite3.connect(path, isolation_level=None)
    try:
        writer.execute("DELETE FROM messages WHERE id = ?", (rows[0],))
        writer.execute(
            "UPDATE messages SET content = ?, fts_content = ? WHERE id = ?",
            ("concurrent canonical update", "concurrent canonical update", rows[1]),
        )
    finally:
        writer.close()

    reopened = SessionDB(db_path=path)
    try:
        assert reopened.optimize_fts_storage(vacuum=False)["ok"] is True
        assert not reopened.search_messages("bounded marker 0")
        assert [row["id"] for row in reopened.search_messages("concurrent canonical update")] == [rows[1]]
        assert not reopened.search_messages("bounded marker 1")
        _strict_integrity_probe(reopened)
    finally:
        reopened.close()


def test_dirty_journal_uses_docsize_for_already_indexed_rows(tmp_path):
    path = tmp_path / "state.db"
    db, rows = _prepare_pending_projection(path)
    db._FTS_REBUILD_CHUNK_ROWS = 32
    db.close()

    writer = sqlite3.connect(path, isolation_level=None)
    try:
        writer.execute(
            "UPDATE messages SET content = ?, fts_content = ? WHERE id = ?",
            ("already indexed replacement", "already indexed replacement", rows[0]),
        )
        writer.execute("DELETE FROM messages WHERE id = ?", (rows[1],))
    finally:
        writer.close()

    reopened = SessionDB(db_path=path)
    try:
        assert reopened._finalize_projection_surfaces() is True
        assert [row["id"] for row in reopened.search_messages("already indexed replacement")] == [rows[0]]
        assert not reopened.search_messages("bounded marker 0")
        assert not reopened.search_messages("bounded marker 1")
        assert _strict_integrity_probe(reopened) is None
    finally:
        reopened.close()


def test_insert_then_update_is_replayed_before_forward_scan(tmp_path):
    db, rows = _prepare_pending_projection(tmp_path / "state.db")
    try:
        new_id = db.append_message("s", role="user", content="inserted before scan")
        db._conn.execute(
            "UPDATE messages SET content = ?, fts_content = ? WHERE id = ?",
            ("updated before scan", "updated before scan", new_id),
        )
        db._conn.execute(
            "DELETE FROM fts_projection_dirty WHERE message_id < ?", (new_id,)
        )
        db._conn.commit()
        db._FTS_REBUILD_CHUNK_ROWS = 1
        assert db._projection_finalize_step() is True
        assert db._conn.execute(
            "SELECT content FROM messages_fts WHERE rowid = ?", (new_id,)
        ).fetchone()[0] == "updated before scan"
        assert db._conn.execute(
            "SELECT 1 FROM fts_projection_dirty WHERE message_id = ?", (new_id,)
        ).fetchone() is None
    finally:
        db.close()


def test_projection_resume_does_not_remark_published_surface(tmp_path):
    db, _rows = _prepare_pending_projection(tmp_path / "state.db")
    try:
        db._conn.execute(
            "INSERT INTO state_meta(key, value) VALUES('fts_projection_started', '1') "
            "ON CONFLICT(key) DO UPDATE SET value = '1'"
        )
        db._conn.execute(
            "INSERT INTO state_meta(key, value) VALUES('fts_projection_surfaces', 'base,trigram') "
            "ON CONFLICT(key) DO UPDATE SET value = 'base,trigram'"
        )
        db._conn.execute(
            "INSERT INTO state_meta(key, value) VALUES('fts_projection_progress', '99') "
            "ON CONFLICT(key) DO UPDATE SET value = '99'"
        )
        db._conn.execute(
            "DELETE FROM state_meta WHERE key = ?", (FTS_PROJECTION_PENDING_KEY,)
        )
        db._conn.commit()

        db._mark_projection_rebuilds_for_backfill()

        assert db.get_meta(FTS_PROJECTION_PENDING_KEY) is None
        assert db.get_meta(FTS_TRIGRAM_PROJECTION_PENDING_KEY) == "1"
        assert db.get_meta("fts_projection_surfaces") == "trigram"
        assert db.get_meta("fts_projection_progress") == "0"
    finally:
        db.close()


def test_session_eligibility_change_replays_existing_trigram_row(tmp_path):
    db, rows = _prepare_pending_projection(tmp_path / "state.db")
    try:
        db._conn.execute(
            "UPDATE sessions SET source = 'cron' WHERE id = ?", ("s",)
        )
        db._conn.commit()
        assert db._projection_finalize_step() is True
        assert db._conn.execute(
            "SELECT 1 FROM messages_fts_trigram_docsize WHERE id = ?", (rows[0],)
        ).fetchone() is None
        assert db._conn.execute(
            "SELECT 1 FROM messages_fts_docsize WHERE id = ?", (rows[0],)
        ).fetchone() is not None
    finally:
        db.close()


def test_projection_finalization_keeps_raw_replay_separate_from_text_index(tmp_path):
    db, rows = _prepare_pending_projection(tmp_path / "state.db")
    try:
        assert db._finalize_projection_surfaces() is True
        row = db._conn.execute(
            "SELECT content, fts_content FROM messages WHERE id = ?", (rows[2],)
        ).fetchone()
        indexed = db._conn.execute(
            "SELECT content FROM messages_fts WHERE rowid = ?", (rows[2],)
        ).fetchone()[0]
        assert "RAW2" in row["content"]
        assert row["fts_content"] == "bounded marker 2"
        assert indexed == row["fts_content"]
        assert "RAW2" not in indexed
        _strict_integrity_probe(db)
    finally:
        db.close()


def test_unavailable_optional_debt_is_bounded_and_does_not_block_base_publish(tmp_path, monkeypatch):
    """Unavailable trigram debt must not head-of-line block available base work."""
    db, _rows = _prepare_pending_projection(tmp_path / "state.db")
    try:
        db._FTS_REBUILD_CHUNK_ROWS = 1
        monkeypatch.setattr(db, "_trigram_tokenizer_is_loadable", lambda conn: False)

        # The first replay leaves this row waiting only for trigram.  A later
        # base-actionable row must still be selected on the next bounded step.
        assert db._projection_finalize_step() is True
        first_flags = db._conn.execute(
            "SELECT base_replayed, trigram_replayed FROM fts_projection_dirty "
            "ORDER BY message_id LIMIT 1"
        ).fetchone()
        assert tuple(first_flags) == (1, 0)

        steps = [db._projection_finalize_step() for _ in range(16)]
        assert False in steps, "bounded finalization must reach its range boundary"
        published, base_pending = db._projection_publish()
        assert published is True
        assert base_pending is False
        assert db.get_meta(FTS_PROJECTION_PENDING_KEY) is None
        assert db.get_meta(FTS_TRIGRAM_PROJECTION_PENDING_KEY) == "1"
        assert db._projection_has_actionable_dirty(db._conn) is False
        assert _trigger_names(db, _FTS_TRIGRAM_TRIGGERS) == set()
    finally:
        db.close()


def test_dirty_replay_restarts_after_second_update_before_base_publication(tmp_path, monkeypatch):
    """A second canonical update must reset a partially replayed dirty row."""
    db, rows = _prepare_pending_projection(tmp_path / "state.db")
    try:
        db._FTS_REBUILD_CHUNK_ROWS = 1
        monkeypatch.setattr(db, "_trigram_tokenizer_is_loadable", lambda conn: False)
        assert db._projection_finalize_step() is True

        db._conn.execute(
            "UPDATE messages SET content = ?, fts_content = ? WHERE id = ?",
            ("second canonical update", "second canonical update", rows[0]),
        )
        db._conn.commit()
        flags = db._conn.execute(
            "SELECT base_replayed, trigram_replayed FROM fts_projection_dirty WHERE message_id = ?",
            (rows[0],),
        ).fetchone()
        assert tuple(flags) == (0, 0)

        steps = [db._projection_finalize_step() for _ in range(20)]
        assert False in steps
        published, base_pending = db._projection_publish()
        assert published is True
        assert base_pending is False
        assert [row["id"] for row in db.search_messages("second canonical update")] == [rows[0]]
        assert not db.search_messages("bounded marker 0")
        db._conn.execute(
            "INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)"
        )
    finally:
        db.close()


def test_optional_debt_reopens_and_finishes_without_duplicate_base_publication(tmp_path, monkeypatch):
    """Restored trigram capability drains retained debt while base stays live."""
    path = tmp_path / "state.db"
    db, rows = _prepare_pending_projection(path)
    monkeypatch.setattr(db, "_trigram_tokenizer_is_loadable", lambda conn: False)
    try:
        db._FTS_REBUILD_CHUNK_ROWS = 1
        for _ in range(24):
            if not db._projection_finalize_step():
                break
        published, base_pending = db._projection_publish()
        assert published is True
        assert base_pending is False
        assert db.get_meta(FTS_PROJECTION_PENDING_KEY) is None
        assert db.get_meta(FTS_TRIGRAM_PROJECTION_PENDING_KEY) == "1"

        db._conn.execute(
            "UPDATE messages SET content = ?, fts_content = ? WHERE id = ?",
            ("live base trigger update", "live base trigger update", rows[0]),
        )
        db._conn.commit()
        assert [row["id"] for row in db.search_messages("live base trigger update")] == [rows[0]]
        assert db._conn.execute(
            "SELECT rowid FROM messages_fts_trigram WHERE messages_fts_trigram MATCH ?",
            ("bounded",),
        ).fetchone() is None
        base_rows_before_reopen = db._conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE rowid = ?", (rows[0],)
        ).fetchone()[0]
    finally:
        db.close()

    reopened = SessionDB(db_path=path)
    try:
        assert reopened.get_meta(FTS_TRIGRAM_PROJECTION_PENDING_KEY) == "1"
        assert reopened.optimize_fts_storage(vacuum=False)["ok"] is True
        assert reopened.get_meta(FTS_PROJECTION_PENDING_KEY) is None
        assert reopened.get_meta(FTS_TRIGRAM_PROJECTION_PENDING_KEY) is None
        assert reopened._conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE rowid = ?", (rows[0],)
        ).fetchone()[0] == base_rows_before_reopen
        assert reopened._conn.execute(
            "SELECT content FROM messages_fts WHERE rowid = ?", (rows[0],)
        ).fetchone()[0] == "live base trigger update"
        assert reopened._conn.execute(
            "SELECT content FROM messages_fts_trigram WHERE rowid = ?", (rows[0],)
        ).fetchone()[0] == "live base trigger update"
        reopened._conn.execute(
            "INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)"
        )
        reopened._conn.execute(
            "INSERT INTO messages_fts_trigram(messages_fts_trigram, rank) VALUES('integrity-check', 1)"
        )
    finally:
        reopened.close()
