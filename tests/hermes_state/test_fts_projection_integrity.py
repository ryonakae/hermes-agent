"""Regression for the ``messages_fts`` external-content drift (issue #114169).

The base word index used to read its external content from the raw ``messages``
table while the triggers indexed only a bounded prefix of every long tool row
and the boundary itself moved with a ``state_meta`` high-water marker. FTS5's
strict ``rank=1`` integrity check re-reads the content source and compares it
with the stored token stream, so that construction could not stay consistent: a
long tool row was enough to make the check report ``fts5: checksum mismatch``,
and the delete/update commands sent full content for a truncated row, leaving
index tokens behind.

These are behaviour contracts on the projection, not on its shape: the strict
probe must survive arbitrary churn, and an index left reading raw ``messages``
must realign once, on open.
"""

import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_common import FTS_PROJECTION_PENDING_KEY, FTS_STORAGE_VERSION, FTS_TOOL_CONTENT_PREFIX_CHARS, _FTS_TRIGGERS

LONG_TOOL_ROW = "prefix " + ("padding " * ((FTS_TOOL_CONTENT_PREFIX_CHARS // 8) + 64)) + " tailtoken"


def _strict_integrity_probe(db) -> None:
    """FTS5's strict check: re-reads the content source and compares it with the
    stored token stream. Raises ``sqlite3.DatabaseError`` when they disagree."""
    db._conn.execute(
        "INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)"
    )


@pytest.fixture
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    if not session_db._fts_enabled:
        session_db.close()
        pytest.skip("SQLite FTS5 unavailable")
    session_db.create_session("session", source="cli")
    try:
        yield session_db
    finally:
        session_db.close()


def test_strict_integrity_probe_survives_tool_row_churn(db):
    """Every write path must leave the index readable by FTS5's own checker.

    Pre-fix, the long-row assertion raises ``DatabaseError: fts5: checksum
    mismatch for table "messages_fts"``: the row was indexed as a prefix while
    ``content='messages'`` kept the full body, so the checker disagreed with the
    stored tokens. The UPDATE and DELETE legs are the triggers whose content
    command used to be re-evaluated against a moving mark.
    """
    _strict_integrity_probe(db)

    long_id = db.append_message(
        "session", role="tool", content=LONG_TOOL_ROW, tool_name="terminal"
    )
    _strict_integrity_probe(db)

    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE messages SET content = ? WHERE id = ?", ("short body", long_id)
        )
    )
    _strict_integrity_probe(db)

    db.append_message("session", role="user", content="short user row")
    _strict_integrity_probe(db)

    db._execute_write(
        lambda conn: conn.execute("DELETE FROM messages WHERE id = ?", (long_id,))
    )
    _strict_integrity_probe(db)


def test_multimodal_content_uses_text_only_fts_projection(db):
    """Stored image parts must not leak into the FTS projection.

    ``messages.content`` keeps the complete multimodal payload for replay, but
    FTS indexes should receive only text parts so search snippets and index
    storage never expose data URLs or image metadata.
    """
    message_id = db.append_message(
        "session",
        role="user",
        content=[
            {"type": "text", "text": "projection marker"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,AAAA"},
            },
        ],
    )

    with db._lock:
        base = db._conn.execute(
            "SELECT content FROM messages_fts WHERE rowid = ?", (message_id,)
        ).fetchone()[0]
        trigram = db._conn.execute(
            "SELECT content FROM messages_fts_trigram WHERE rowid = ?", (message_id,)
        ).fetchone()[0]

    assert base == "projection marker"
    assert trigram == "projection marker"


def test_batch_and_replace_writers_persist_text_projection(db):
    """Every split message writer keeps replay bytes separate from indexed text."""
    inserted = db.append_messages_batch(
        "session",
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "batch projection marker"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,BATCH"}},
                ],
            }
        ],
    )
    assert inserted == 1
    row = db._conn.execute(
        "SELECT content, fts_content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT 1",
        ("session",),
    ).fetchone()
    assert row["content"].startswith("\x00json:")
    assert row["fts_content"] == "batch projection marker"

    replacement = [
        {"type": "text", "text": "replacement projection marker"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,REPLACE"}},
    ]
    db.replace_messages("session", [{"role": "user", "content": replacement}])
    assert db.get_messages("session")[0]["content"] == replacement
    row = db._conn.execute(
        "SELECT id, content, fts_content FROM messages WHERE session_id = ?", ("session",)
    ).fetchone()
    assert "REPLACE" in row["content"]
    assert row["fts_content"] == "replacement projection marker"
    assert db._conn.execute(
        "SELECT content FROM messages_fts WHERE rowid = ?", (row["id"],)
    ).fetchone()[0] == "replacement projection marker"
    assert not db.search_messages("batch")
    _strict_integrity_probe(db)


def test_old_projection_views_rebuild_on_optimize(tmp_path):
    """Pre-projection external-content views are rebuilt by deferred optimize."""
    path = tmp_path / "state.db"
    first = SessionDB(db_path=path)
    if not first._fts_enabled or not first._trigram_available:
        first.close()
        pytest.skip("SQLite FTS5 trigram unavailable")
    first.create_session("session", source="cli")
    message_id = first.append_message(
        "session",
        role="user",
        content=[
            {"type": "text", "text": "reopen projection marker"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,REOPEN"}},
        ],
    )
    with first._lock:
        for trigger in (
            "messages_fts_insert", "messages_fts_delete", "messages_fts_update",
            "messages_fts_trigram_insert", "messages_fts_trigram_delete", "messages_fts_trigram_update",
        ):
            first._conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        first._conn.execute("DROP TABLE messages_fts")
        first._conn.execute("DROP VIEW messages_fts_src")
        first._conn.execute(
            "CREATE VIEW messages_fts_src AS SELECT id, content, tool_name, tool_calls FROM messages"
        )
        first._conn.execute(
            "CREATE VIRTUAL TABLE messages_fts USING fts5(content, tool_name, tool_calls, "
            "content='messages_fts_src', content_rowid='id')"
        )
        first._conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
        first._conn.execute("DROP TABLE messages_fts_trigram")
        first._conn.execute("DROP VIEW messages_fts_trigram_src")
        first._conn.execute(
            "CREATE VIEW messages_fts_trigram_src AS "
            "SELECT id, role, content, tool_name FROM messages WHERE role <> 'tool'"
        )
        first._conn.execute(
            "CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(content, tool_name, "
            "content='messages_fts_trigram_src', content_rowid='id', tokenize='trigram')"
        )
        first._conn.execute("INSERT INTO messages_fts_trigram(messages_fts_trigram) VALUES('rebuild')")
        first._conn.commit()
    try:
        assert first.fts_optimize_available() is True
        assert first.optimize_fts_storage(vacuum=False)["ok"] is True
        assert first.get_meta("fts_storage_version") == str(FTS_STORAGE_VERSION)
        assert first._conn.execute(
            "SELECT content FROM messages_fts WHERE rowid = ?", (message_id,)
        ).fetchone()[0] == "reopen projection marker"
        assert first._conn.execute(
            "SELECT content FROM messages_fts_trigram WHERE rowid = ?", (message_id,)
        ).fetchone()[0] == "reopen projection marker"
        for view in ("messages_fts_src", "messages_fts_trigram_src"):
            sql = first._conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'view' AND name = ?", (view,)
            ).fetchone()[0]
            assert "fts_content" in sql
    finally:
        first.close()


def test_projection_backfill_resumes_after_interrupted_open(tmp_path, monkeypatch):
    """A committed row backfill survives interruption and completes on reopen."""
    path = tmp_path / "state.db"
    first = SessionDB(db_path=path)
    first.create_session("session", source="cli")
    first.append_message(
        "session", role="user",
        content=[{"type": "text", "text": "resume marker"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,RESUME"}}],
    )
    conn = sqlite3.connect(path)
    conn.execute("UPDATE messages SET fts_content = NULL")
    conn.execute(
        "INSERT INTO state_meta(key, value) VALUES('fts_optimize_available', '1') "
        "ON CONFLICT(key) DO UPDATE SET value = '1'"
    )
    conn.commit()
    conn.close()

    from hermes_state_schema import SessionSchemaMixin
    original = SessionSchemaMixin._backfill_fts_content
    interrupted = {"done": False}

    def interrupt(cursor, after_id=0, limit=500):
        row = cursor.execute(
            "SELECT id, content FROM messages WHERE id > ? AND fts_content IS NULL LIMIT 1", (after_id,)
        ).fetchone()
        if row is not None:
            cursor.execute("UPDATE messages SET fts_content = ? WHERE id = ?", ("resume marker", row[0]))
            interrupted["done"] = True
            raise RuntimeError("simulated projection interruption")
        return original(cursor, after_id, limit)

    monkeypatch.setattr(SessionSchemaMixin, "_backfill_fts_content", staticmethod(interrupt))
    interrupted_db = SessionDB(db_path=path)
    with pytest.raises(RuntimeError, match="simulated projection interruption"):
        interrupted_db.optimize_fts_storage(vacuum=False)
    interrupted_db.close()
    monkeypatch.setattr(SessionSchemaMixin, "_backfill_fts_content", staticmethod(original))

    resumed = SessionDB(db_path=path)
    try:
        assert resumed.fts_optimize_available() is True
        assert resumed.optimize_fts_storage(vacuum=False)["ok"] is True
        assert resumed.get_meta("fts_optimize_available") is None
        assert resumed._conn.execute("SELECT fts_content FROM messages").fetchone()[0] == "resume marker"
    finally:
        resumed.close()
    assert interrupted["done"]

    monkeypatch.setattr(SessionSchemaMixin, "_backfill_fts_content", staticmethod(original))
    reopened = SessionDB(db_path=path)
    try:
        assert reopened._conn.execute("SELECT fts_content FROM messages").fetchone()[0] == "resume marker"
    finally:
        reopened.close()


def test_index_reading_raw_messages_defers_realign_until_optimize(tmp_path):
    """A populated raw-source index is fenced on open and realigned by optimize."""
    path = tmp_path / "state.db"
    first = SessionDB(db_path=path)
    if not first._fts_enabled:
        first.close()
        pytest.skip("SQLite FTS5 unavailable")
    first.create_session("session", source="cli")
    row_id = first.append_message(
        "session", role="tool", content=LONG_TOOL_ROW, tool_name="terminal"
    )

    # Rewind to the pre-fix on-disk shape: the base index reads raw `messages`,
    # and its tokens were written from the bounded projection the triggers used
    # for tool rows. That mixture IS the drift: the checker re-reads the full
    # body and disagrees with the stored prefix.
    first._conn.execute("DROP TABLE messages_fts")
    first._conn.execute(
        "CREATE VIRTUAL TABLE messages_fts USING fts5("
        "content, tool_name, tool_calls, content='messages', content_rowid='id')"
    )
    first._conn.execute(
        "INSERT INTO messages_fts(rowid, content, tool_name, tool_calls) VALUES(?, ?, ?, ?)",
        (row_id, LONG_TOOL_ROW[:FTS_TOOL_CONTENT_PREFIX_CHARS], "terminal", None),
    )
    first._conn.execute(
        "INSERT INTO state_meta(key, value) VALUES('fts_storage_version', '2') "
        "ON CONFLICT(key) DO UPDATE SET value = '2'"
    )
    first._conn.execute(
        "INSERT OR REPLACE INTO state_meta(key, value) VALUES('fts_tool_full_content_high_water', ?)",
        (str(row_id),),
    )
    with pytest.raises(sqlite3.DatabaseError):
        _strict_integrity_probe(first)
    first.close()

    migrated = SessionDB(db_path=path)
    try:
        assert migrated.get_meta("fts_tool_full_content_high_water") is None
        assert migrated.get_meta(FTS_PROJECTION_PENDING_KEY) == "1"
        assert migrated._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' AND name IN (?, ?, ?)",
            _FTS_TRIGGERS[:3],
        ).fetchall() == []
        index_sql = migrated._conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'messages_fts'"
        ).fetchone()[0]
        assert "messages_fts_src" in index_sql
        # The pending route stays offline; the explicit tool path reads canonical
        # content and remains available while the projection is fenced.
        assert [
            row["id"]
            for row in migrated.search_messages("tailtoken", role_filter=["tool"])
        ] == [row_id]
        assert migrated.optimize_fts_storage(vacuum=False)["ok"] is True
        assert migrated.get_meta(FTS_PROJECTION_PENDING_KEY) is None
    finally:
        migrated.close()
