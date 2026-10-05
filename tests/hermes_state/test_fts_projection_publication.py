"""Publishing a completed projection leaves working canonical writer routes."""

from hermes_state import SessionDB
from hermes_state_common import FTS_TRIGRAM_PROJECTION_PENDING_KEY


def test_projection_chunks_yield_to_other_writers(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    pauses = []
    try:
        db.create_session("s", source="cli")
        for text in ("first", "second", "third"):
            db.append_message("s", "user", text)
        db._FTS_REBUILD_CHUNK_ROWS = 1
        db._mark_projection_rebuilds_for_backfill()
        monkeypatch.setattr("hermes_state_search.time.sleep", pauses.append)
        assert db._finalize_projection_surfaces()
        assert len(pauses) >= 2
        assert all(pause >= db._FTS_REBUILD_MIN_PAUSE for pause in pauses)
    finally:
        db.close()


def test_completed_projection_removes_journal_triggers_before_table(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s", source="cli")
        db.append_message("s", "user", "beforepublication")
        db._mark_projection_rebuilds_for_backfill()
        db._backfill_projection_content()
        assert db._finalize_projection_surfaces()
        db.append_message("s", "user", "afterpublication")
        assert db.search_messages("afterpublication")
        assert not db._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='trigger' "
            "AND name LIKE '%fts_projection_dirty%'"
        ).fetchall()
        db._conn.execute(
            "INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)"
        )
    finally:
        db.close()


def test_unavailable_optional_surface_stays_fenced_at_publication(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s", source="cli")
        db.append_message("s", "user", "optionalpublication")
        db._mark_projection_rebuilds_for_backfill()
        monkeypatch.setattr(db, "_trigram_tokenizer_is_loadable", lambda conn: False)
        assert db._finalize_projection_surfaces()
        assert db.get_meta(FTS_TRIGRAM_PROJECTION_PENDING_KEY) == "1"
        assert not db._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='trigger' "
            "AND name IN ('messages_fts_trigram_insert','messages_fts_trigram_delete','messages_fts_trigram_update')"
        ).fetchall()
        db.append_message("s", "user", "baseavailable")
        assert db.search_messages("baseavailable")
    finally:
        db.close()
