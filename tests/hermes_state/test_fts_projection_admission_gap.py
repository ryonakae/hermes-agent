"""Canonical writes remain recoverable across the admission lock transition."""

import sqlite3

from hermes_state import SessionDB
from hermes_state_common import FTS_PROJECTION_PENDING_KEY


def test_writer_between_fence_commit_and_begin_preserves_canonical_rows(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    conn = db._conn
    assert conn is not None
    try:
        db.create_session("s", source="cli")
        db.append_message("s", role="user", content="beforegapneedle")
        for name in ("messages_fts_insert", "messages_fts_delete", "messages_fts_update"):
            conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        conn.execute("DROP TABLE messages_fts")
        conn.execute("DROP VIEW messages_fts_src")
        conn.execute(
            "CREATE VIEW messages_fts_src AS SELECT id, content, tool_name, tool_calls FROM messages"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE messages_fts USING fts5(content, tool_name, tool_calls, "
            "content='messages_fts_src', content_rowid='id')"
        )
        conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
        conn.commit()
        interleaved = []

        class CommitBoundary:
            def __getattr__(self, name):
                return getattr(conn, name)

            def commit(self):
                conn.commit()
                if interleaved:
                    return
                peer = sqlite3.connect(path, timeout=1)
                try:
                    pending = peer.execute(
                        "SELECT value FROM state_meta WHERE key = ?", (FTS_PROJECTION_PENDING_KEY,)
                    ).fetchone()
                    assert pending is not None and pending[0] == "1"
                    assert not peer.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='trigger' "
                        "AND name IN ('messages_fts_insert','messages_fts_delete','messages_fts_update')"
                    ).fetchall()
                    peer.execute(
                        "INSERT INTO messages(session_id,role,content,fts_content,timestamp) "
                        "VALUES ('s','user','duringgapneedle','duringgapneedle',1)"
                    )
                    peer.commit()
                    interleaved.append(True)
                finally:
                    peer.close()

        monkeypatch.setattr(db, "_conn", CommitBoundary())
        try:
            db._migrate_misaligned_fts_source(conn.cursor(), legacy=False)
        finally:
            db._conn = conn
        conn.commit()
        assert interleaved == [True]
        assert db.get_meta(FTS_PROJECTION_PENDING_KEY) == "1"
        assert [row[0] for row in conn.execute(
            "SELECT content FROM messages ORDER BY id"
        )] == ["beforegapneedle", "duringgapneedle"]
        assert db.search_messages("duringgapneedle")
    finally:
        db._conn = conn
        db.close()
