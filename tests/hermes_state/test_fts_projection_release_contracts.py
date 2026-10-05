"""Release contracts for durable FTS projection migration state."""

import sqlite3
import threading
import time

import pytest

from hermes_state import SessionDB
from hermes_state_common import (
    FTS_CJK_PROJECTION_PENDING_KEY,
    FTS_PROJECTION_PENDING_KEY,
    FTS_STALE_KEY,
    FTS_TRIGRAM_PROJECTION_PENDING_KEY,
    _FTS_CJK_TRIGGERS,
    _FTS_TRIGGERS,
)
from hermes_state_fts import _FTS_TRIGRAM_TRIGGERS


def _install_old_base_surface(db):
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
    conn.commit()


def _install_old_trigram_surface(db):
    conn = db._conn
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


def _install_trigger(db, name):
    db._conn.execute(f"DROP TRIGGER IF EXISTS {name}")
    db._conn.execute(
        f"CREATE TRIGGER {name} AFTER INSERT ON messages BEGIN SELECT 1; END"
    )
    db._conn.commit()


def _trigger_names(db, names):
    placeholders = ",".join("?" for _ in names)
    return {
        row[0]
        for row in db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name IN ("
            f"{placeholders})",
            tuple(names),
        )
    }


def test_projection_ddl_crash_reopens_with_durable_pending_and_resumes(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    db.create_session("s", source="cli")
    db.append_message(
        "s",
        role="user",
        content=[{"type": "text", "text": "durable projection needle"}],
    )
    db._conn.execute("UPDATE messages SET fts_content = NULL")
    _install_old_base_surface(db)

    original = db._run_admitted_startup_rebuild

    def crash_after_durable_swap(cursor, rebuild_fn):
        original(cursor, rebuild_fn)
        raise RuntimeError("simulated projection DDL crash")

    monkeypatch.setattr(db, "_run_admitted_startup_rebuild", crash_after_durable_swap)
    with pytest.raises(RuntimeError, match="projection DDL crash"):
        db._migrate_misaligned_fts_source(db._conn, legacy=False)
    db.close()

    reopened = SessionDB(db_path=path)
    try:
        assert reopened.get_meta(FTS_PROJECTION_PENDING_KEY) == "1"
        assert reopened.get_meta("fts_storage_version") != "4"
        assert _trigger_names(reopened, _FTS_TRIGGERS) == set()
        assert reopened.search_messages("durable projection needle")
        assert reopened.optimize_fts_storage(vacuum=False)["ok"] is True
        assert reopened.get_meta(FTS_PROJECTION_PENDING_KEY) is None
        assert reopened.search_messages("durable projection needle")
    finally:
        reopened.close()


def test_tokenizerless_trigram_surface_is_quarantined_without_pending_marker(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s", source="cli")
        db.append_message("s", role="user", content="legacy trigram surface")
        _install_old_trigram_surface(db)
        old_view = db._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='view' AND name='messages_fts_trigram_src'"
        ).fetchone()[0]
        monkeypatch.setattr(db, "_trigram_tokenizer_is_loadable", lambda _conn: False)

        db._migrate_trigram_projection_source(db._conn)
        db._conn.commit()

        assert db._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='view' AND name='messages_fts_trigram_src'"
        ).fetchone()[0] == old_view
        assert db.get_meta(FTS_TRIGRAM_PROJECTION_PENDING_KEY) == "1"
        assert _trigger_names(db, _FTS_TRIGRAM_TRIGGERS) == set()
        assert db._trigram_available is False
    finally:
        db.close()


def test_tokenizerless_old_trigram_view_drops_writer_triggers(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s", source="cli")
        db.append_message("s", role="user", content="legacy trigram surface")
        _install_old_trigram_surface(db)
        for name in _FTS_TRIGRAM_TRIGGERS:
            _install_trigger(db, name)
        monkeypatch.setattr(db, "_trigram_tokenizer_is_loadable", lambda _conn: False)

        db._migrate_trigram_projection_source(db._conn)
        db._conn.commit()

        assert db.get_meta(FTS_TRIGRAM_PROJECTION_PENDING_KEY) is None
        assert _trigger_names(db, _FTS_TRIGRAM_TRIGGERS) == set()
        old_view = db._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='view' AND name='messages_fts_trigram_src'"
        ).fetchone()[0]
        assert "fts_content" not in old_view.lower()
    finally:
        db.close()


def test_cjk_residual_triggers_are_removed_when_table_is_absent(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        for name in _FTS_CJK_TRIGGERS:
            _install_trigger(db, name)
        db._fts_cjk_loaded = False
        db._ensure_fts_cjk_schema(db._conn)
        db._conn.commit()
        assert _trigger_names(db, _FTS_CJK_TRIGGERS) == set()
    finally:
        db.close()


def test_cjk_trigger_quarantine_failure_is_not_swallowed(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        for name in _FTS_CJK_TRIGGERS:
            _install_trigger(db, name)
        db._fts_cjk_loaded = False
        db._conn.set_authorizer(
            lambda action, *_args: sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_DROP_TRIGGER else sqlite3.SQLITE_OK
        )
        with pytest.raises(sqlite3.Error, match="quarantine|denied|not authorized"):
            db._ensure_fts_cjk_schema(db._conn)
    finally:
        db._conn.set_authorizer(None)
        db.close()


def test_fts5_unavailable_startup_quarantines_cjk_and_all_writer_triggers(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        for name in _FTS_TRIGGERS + _FTS_CJK_TRIGGERS:
            _install_trigger(db, name)
        monkeypatch.setattr(db, "_sqlite_supports_fts5", lambda _cursor: False)
        db._init_schema()
        db._conn.commit()
        assert _trigger_names(db, _FTS_TRIGGERS + _FTS_CJK_TRIGGERS) == set()
    finally:
        db.close()


def test_drop_all_fts_triggers_fails_closed_when_drop_is_denied(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        surviving = _FTS_TRIGGERS[0]
        authorizer = db._conn.set_authorizer
        authorizer(lambda action, *_args: sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_DROP_TRIGGER else sqlite3.SQLITE_OK)
        with pytest.raises(sqlite3.Error, match="quarantine|denied|not authorized"):
            db._drop_all_fts_triggers(db._conn.cursor())
        assert surviving in _trigger_names(db, _FTS_TRIGGERS)
    finally:
        db._conn.set_authorizer(None)
        db.close()


def test_stale_recovery_does_not_restore_pending_trigram(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db._conn.execute(
            "INSERT INTO state_meta(key, value) VALUES(?, '1')",
            (FTS_TRIGRAM_PROJECTION_PENDING_KEY,),
        )
        db._conn.execute(
            "INSERT INTO state_meta(key, value) VALUES(?, '1')",
            (FTS_STALE_KEY,),
        )
        db._conn.execute(
            "CREATE VIEW IF NOT EXISTS messages_fts_trigram_src AS "
            "SELECT id, role, content, tool_name FROM messages WHERE role <> 'tool'"
        )
        db._conn.commit()
        db._recover_stale_fts_locked(db._conn.cursor(), legacy=False)
        assert db._trigram_available is False
        assert _trigger_names(db, _FTS_TRIGRAM_TRIGGERS) == set()
        assert db.get_meta(FTS_TRIGRAM_PROJECTION_PENDING_KEY) == "1"
        assert db._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='view' AND name='messages_fts_trigram_src'"
        ).fetchone() is None
    finally:
        db.close()


def test_optimize_does_not_demote_while_projection_is_pending(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db._conn.execute(
            "INSERT INTO state_meta(key, value) VALUES(?, '1')",
            (FTS_TRIGRAM_PROJECTION_PENDING_KEY,),
        )
        db._conn.commit()
        called = []
        monkeypatch.setattr(db, "_demote_legacy_fts_to_trash", lambda: called.append(True))
        monkeypatch.setattr(db, "_upgrade_projection_surfaces", lambda: None)
        db.optimize_fts_storage(vacuum=False)
        assert called == []
    finally:
        db.close()


def test_open_peer_quarantines_pending_surface_and_recovers_after_finalize(tmp_path):
    path = tmp_path / "state.db"
    owner = SessionDB(db_path=path)
    peer = SessionDB(db_path=path)
    try:
        owner._conn.execute(
            "INSERT INTO state_meta(key, value) VALUES(?, '1')",
            (FTS_TRIGRAM_PROJECTION_PENDING_KEY,),
        )
        owner._conn.execute("DROP TRIGGER IF EXISTS messages_fts_trigram_insert")
        owner._conn.execute("DROP TRIGGER IF EXISTS messages_fts_trigram_delete")
        owner._conn.execute("DROP TRIGGER IF EXISTS messages_fts_trigram_update")
        owner._conn.commit()
        peer._refresh_fts_stale_state()
        assert peer._trigram_available is False
        assert peer._fts_stale is True

        owner._conn.execute(
            "DELETE FROM state_meta WHERE key = ?", (FTS_TRIGRAM_PROJECTION_PENDING_KEY,)
        )
        owner._conn.executescript(
            "CREATE TRIGGER messages_fts_trigram_insert AFTER INSERT ON messages "
            "WHEN new.role <> 'tool' BEGIN INSERT INTO messages_fts_trigram(rowid, content, tool_name) "
            "VALUES(new.id, new.content, new.tool_name); END;"
        )
        owner._conn.executescript(
            "CREATE TRIGGER messages_fts_trigram_delete AFTER DELETE ON messages "
            "WHEN old.role <> 'tool' BEGIN SELECT 1; END;"
        )
        owner._conn.executescript(
            "CREATE TRIGGER messages_fts_trigram_update AFTER UPDATE OF content ON messages "
            "BEGIN SELECT 1; END;"
        )
        owner._conn.commit()
        peer._refresh_fts_stale_state()
        assert peer._fts_stale is False
        assert peer._trigram_available is True
    finally:
        peer.close()
        owner.close()


def test_projection_rebuild_holds_sqlite_write_fence(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def rebuild(_cursor):
        entered.set()
        assert release.wait(5)

    def run():
        try:
            db._run_admitted_startup_rebuild(db._conn.cursor(), lambda: rebuild(db._conn.cursor()))
        finally:
            finished.set()

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert entered.wait(5)
        sibling = sqlite3.connect(str(db.db_path), timeout=0.0, isolation_level=None)
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                sibling.execute("BEGIN IMMEDIATE")
        finally:
            sibling.close()
        release.set()
        assert finished.wait(5)
    finally:
        release.set()
        thread.join(timeout=5)
        db.close()
    assert not thread.is_alive()
