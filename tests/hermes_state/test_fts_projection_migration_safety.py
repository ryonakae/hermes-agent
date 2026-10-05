"""Focused contracts for projection migration fencing and quarantine."""

import pytest

from hermes_state import SessionDB
from hermes_state_common import FTS_CJK_PROJECTION_PENDING_KEY, FTS_PROJECTION_PENDING_KEY
from hermes_state_fts import FTS_CJK_TRIGGER_SQL


def _install_old_base_surface(db):
    conn = db._conn
    for name in ("messages_fts_insert", "messages_fts_delete", "messages_fts_update"):
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
    conn.execute("DROP TABLE messages_fts")
    conn.execute("DROP VIEW messages_fts_src")
    conn.execute(
        "CREATE VIEW messages_fts_src AS "
        "SELECT id, content, tool_name, tool_calls FROM messages"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE messages_fts USING fts5("
        "content, tool_name, tool_calls, content='messages_fts_src', content_rowid='id')"
    )
    conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
    conn.commit()


def _install_old_trigram_surface(db):
    conn = db._conn
    for name in (
        "messages_fts_trigram_insert",
        "messages_fts_trigram_delete",
        "messages_fts_trigram_update",
    ):
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
    conn.execute("DROP TABLE messages_fts_trigram")
    conn.execute("DROP VIEW messages_fts_trigram_src")
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


def test_pending_projection_fences_base_writers_before_rebuild(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s", source="cli")
        db.append_message("s", role="user", content="pending fence")
        _install_old_base_surface(db)

        def crash(*_args):
            raise RuntimeError("simulated projection crash")

        monkeypatch.setattr(db, "_run_admitted_startup_rebuild", crash)
        with pytest.raises(RuntimeError, match="simulated projection crash"):
            db._migrate_misaligned_fts_source(db._conn, legacy=False)

        assert db.get_meta(FTS_PROJECTION_PENDING_KEY) == "1"
        assert db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' "
            "AND name IN ('messages_fts_insert', 'messages_fts_delete', 'messages_fts_update')"
        ).fetchall() == []
    finally:
        db.close()


def test_tokenizerless_trigram_migration_preserves_existing_surface(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s", source="cli")
        db.append_message("s", role="user", content="legacy trigram surface")
        _install_old_trigram_surface(db)
        old_view = db._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='view' AND name='messages_fts_trigram_src'"
        ).fetchone()[0]
        monkeypatch.setattr(db, "_trigram_tokenizer_is_loadable", lambda _conn: False, raising=False)

        db._migrate_trigram_projection_source(db._conn)

        assert db._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='view' AND name='messages_fts_trigram_src'"
        ).fetchone()[0] == old_view
        # A tokenizerless runtime leaves the old readable view in place but
        # records durable migration debt for a capable reopen.
        assert db.get_meta("fts_trigram_projection_rebuild_pending") == "1"
    finally:
        db.close()


def test_cjk_projection_update_trigger_includes_durable_text_projection():
    update = FTS_CJK_TRIGGER_SQL.upper()
    assert "AFTER UPDATE OF CONTENT, FTS_CONTENT, TOOL_NAME, TOOL_CALLS, ROLE" in update
    assert "OLD.FTS_CONTENT IS NOT NEW.FTS_CONTENT" in update
    assert FTS_CJK_PROJECTION_PENDING_KEY not in update
