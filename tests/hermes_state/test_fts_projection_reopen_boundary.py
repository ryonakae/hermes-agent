"""Old external-content stores must remain usable across projection migration."""

import pytest

from hermes_state import SessionDB
from hermes_state_common import FTS_PROJECTION_PENDING_KEY, _FTS_TRIGGERS
from hermes_state_schema import SessionSchemaMixin

_FTS_BASE_TRIGGERS = tuple(name for name in _FTS_TRIGGERS if "_trigram_" not in name)


@pytest.mark.parametrize("populated", [False, True], ids=["empty", "historical-multimodal"])
def test_old_base_projection_survives_reopen(tmp_path, populated):
    path = tmp_path / "state.db"
    original = SessionDB(db_path=path)
    content = [{"type": "text", "text": "historicalneedle"},
               {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    try:
        if populated:
            original.create_session("s", source="cli")
            original.append_message("s", role="user", content=content)
            original._conn.execute("UPDATE messages SET fts_content = NULL")
        for name in ("messages_fts_insert", "messages_fts_delete", "messages_fts_update"):
            original._conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        original._conn.execute("DROP TABLE messages_fts")
        original._conn.execute("DROP VIEW messages_fts_src")
        original._conn.execute(
            "CREATE VIEW messages_fts_src AS SELECT id, content, tool_name, tool_calls FROM messages"
        )
        original._conn.execute(
            "CREATE VIRTUAL TABLE messages_fts USING fts5("
            "content, tool_name, tool_calls, content='messages_fts_src', content_rowid='id')"
        )
        original._conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
        original._conn.commit()
    finally:
        original.close()

    reopened = SessionDB(db_path=path)
    try:
        if populated:
            assert reopened.get_messages("s")[0]["content"] == content
            assert reopened.search_messages("historicalneedle"), "reopen must not silently lose historical search"
        else:
            reopened.create_session("after", source="cli")
            reopened.append_message("after", role="user", content="afterneedle")
            assert reopened.search_messages("afterneedle")
        pending = reopened.get_meta(FTS_PROJECTION_PENDING_KEY) == "1"
        if pending:
            assert reopened.optimize_fts_storage(vacuum=False)["ok"] is True
        reopened._conn.execute(
            "INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)"
        )
        if pending:
            assert reopened.get_meta(FTS_PROJECTION_PENDING_KEY) is None
            assert reopened.get_meta("fts_rebuild_high_water") is None
            assert reopened.get_meta("fts_rebuild_progress") is None
            reopened.append_message("s", role="user", content="postmigrationneedle")
            reopened._conn.execute(
                "INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)"
            )
            assert reopened.search_messages("postmigrationneedle")
    finally:
        reopened.close()


def test_populated_old_projection_reopen_defers_full_rebuild_and_falls_back(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    original = SessionDB(db_path=path)
    marker = "startup projection fallback needle"
    content = [{"type": "text", "text": marker},
               {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    try:
        original.create_session("s", source="cli")
        original.append_message("s", role="user", content=content)
        original._conn.execute("UPDATE messages SET fts_content = NULL")
        for name in ("messages_fts_insert", "messages_fts_delete", "messages_fts_update"):
            original._conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        original._conn.execute("DROP TABLE messages_fts")
        original._conn.execute("DROP VIEW messages_fts_src")
        original._conn.execute(
            "CREATE VIEW messages_fts_src AS SELECT id, content, tool_name, tool_calls FROM messages"
        )
        original._conn.execute(
            "CREATE VIRTUAL TABLE messages_fts USING fts5("
            "content, tool_name, tool_calls, content='messages_fts_src', content_rowid='id')"
        )
        original._conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
        original._conn.commit()
    finally:
        original.close()

    rebuild_sql = []
    real_admission = SessionSchemaMixin._run_admitted_startup_rebuild

    def trace_startup_rebuild(self, cursor, rebuild_fn):
        statements = []
        cursor.connection.set_trace_callback(statements.append)
        try:
            return real_admission(self, cursor, rebuild_fn)
        finally:
            cursor.connection.set_trace_callback(None)
            rebuild_sql.extend(statements)

    monkeypatch.setattr(SessionSchemaMixin, "_run_admitted_startup_rebuild", trace_startup_rebuild)
    reopened = SessionDB(db_path=path)
    try:
        assert not any("messages_fts(messages_fts) values('rebuild')" in sql.lower() for sql in rebuild_sql)
        assert reopened.get_meta(FTS_PROJECTION_PENDING_KEY) == "1"
        assert reopened._fts_stale is True
        assert reopened._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' AND name IN (?, ?, ?)",
            _FTS_BASE_TRIGGERS,
        ).fetchall() == []
        assert [row["id"] for row in reopened.search_messages(marker)]
    finally:
        reopened.close()
