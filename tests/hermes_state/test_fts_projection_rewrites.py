"""Text projections follow in-place transcript rewrites without altering replay."""
import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("writer", ["user_rewrite", "assistant_repair"])
def test_in_place_rewrite_updates_multimodal_projection(tmp_path, writer):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("session", source="cli")
        role = "user" if writer == "user_rewrite" else "assistant"
        row_id = db.append_message("session", role=role, content="")
        content = [
            {"type": "text", "text": "rewritten searchable marker"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,PRIVATE"}},
        ]
        if writer == "user_rewrite":
            assert db.set_user_message_content("session", row_id, content) == 1
        else:
            db.append_messages_batch("session", [{
                "role": "assistant", "content": content, "_row_id": row_id,
            }])
        row = db._conn.execute(
            "SELECT content, fts_content FROM messages WHERE id = ?", (row_id,)
        ).fetchone()
        assert row["fts_content"] == "rewritten searchable marker"
        assert "PRIVATE" in row["content"]
        assert db.get_messages("session")[0]["content"] == content
        assert [r["id"] for r in db.search_messages("searchable")] == [row_id]
        assert not db.search_messages("PRIVATE")
        db._conn.execute("INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)")
    finally:
        db.close()
