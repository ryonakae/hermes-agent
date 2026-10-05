"""Projection backfill bounds scanned rows, not only matching legacy rows."""

from hermes_state import SessionDB


def test_sparse_projection_backfill_advances_only_one_id_window(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s", source="cli")
        ids = [db.append_message("s", "user", "already projected") for _ in range(6)]
        last = db.append_message("s", "user", [{"type": "text", "text": "legacyprojection"}])
        conn = db._conn
        assert conn is not None
        conn.execute("UPDATE messages SET fts_content=NULL WHERE id=?", (last,))
        conn.commit()
        cursor = db._execute_write(lambda c: db._backfill_fts_content(c, 0, limit=2))
        assert cursor == ids[1]
        assert conn.execute("SELECT fts_content FROM messages WHERE id=?", (last,)).fetchone()[0] is None
        while cursor is not None:
            cursor = db._execute_write(lambda c: db._backfill_fts_content(c, cursor, limit=2))
        assert conn.execute("SELECT fts_content FROM messages WHERE id=?", (last,)).fetchone()[0] == "legacyprojection"
    finally:
        db.close()
