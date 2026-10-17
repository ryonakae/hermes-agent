"""Durable logical-message identity for unaddressed live-replay rewrites (#129065).

A live replay may intentionally omit the physical SQLite row id. If that dict is edited in
place, the write path must update the logical message it came from without guessing from its
now-mutated payload. The durable message_uid identifies the logical message and the stored-row
snapshot remains the compare-and-swap proof for the version we actually loaded.
"""

from __future__ import annotations

import pytest

from agent.context_compressor import _DB_PERSISTED_MARKER
from agent.message_metadata import DB_ROW_SNAPSHOT, MESSAGE_UID
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    store = SessionDB(tmp_path / "state.db")
    try:
        yield store
    finally:
        store.close()


def _active_rows(store, sid):
    return [
        dict(row)
        for row in store._conn.execute(
            "SELECT id, role, content, timestamp, token_count, message_uid, tool_calls "
            "FROM messages WHERE session_id = ? AND active = 1 ORDER BY id",
            (sid,),
        ).fetchall()
    ]


@pytest.mark.parametrize(
    ("include_row_ids", "content", "token_count"),
    [(False, "after repair", 42), (True, "before repair", 42)],
    ids=["unaddressed", "row-addressed"],
)
def test_mutated_replay_rewrites_its_original_row(db, include_row_ids, content, token_count):
    sid = "mutated-replay"
    db.create_session(sid, "desktop")
    original = {"role": "assistant", "content": "before repair", "timestamp": 1000.0, "token_count": 42}
    assert db.append_messages_batch(sid, [original]) == 1
    before = _active_rows(db, sid)

    # Live replay without physical row ids is a supported shape (gateway/ACP and identity-losing
    # handoffs). It still carries logical identity and, for a writable replay, the CAS version.
    # Row-addressed resume loaders keep the legacy (no-digest) path: the projection never decodes
    # token_count, so a digest-matched rewrite of a resumed row would null it.
    restored = db.get_messages_as_conversation(sid, repair_alternation=True, include_row_ids=include_row_ids)
    assert restored[0][MESSAGE_UID] == before[0]["message_uid"]
    assert (DB_ROW_SNAPSHOT in restored[0]) is not include_row_ids

    # Real rewrite sites change the payload BEFORE clearing the persisted marker. The agent flush row
    # (session_persistence._db_flush_row) always carries the token_count key, None for a replay.
    restored[0]["content"] = "after repair"
    restored[0]["token_count"] = None
    restored[0].pop(_DB_PERSISTED_MARKER, None)

    assert db.append_messages_batch(sid, restored) == 0
    after = _active_rows(db, sid)
    assert len(after) == 1
    assert after[0]["id"] == before[0]["id"]
    assert after[0]["message_uid"] == before[0]["message_uid"]
    # The replay never decodes token_count: the stored value survives either path. A row-addressed
    # (no-digest) resume adopts the filled row's content instead of rewriting it.
    assert (after[0]["content"], after[0]["token_count"]) == (content, token_count)


def test_fresh_identical_message_is_never_adopted_by_payload(db):
    """Content, timestamps and even a copied message_uid are not proof: only uid + CAS snapshot is."""
    sid = "identical-twins"
    db.create_session(sid, "desktop")
    first = {"role": "user", "content": "same", "timestamp": 3000.0}
    assert db.append_messages_batch(sid, [first]) == 1
    first_uid = _active_rows(db, sid)[0]["message_uid"]

    second = {"role": "user", "content": "same", "timestamp": 3000.0, MESSAGE_UID: first_uid}
    assert db.append_messages_batch(sid, [second]) == 1

    assert len(_active_rows(db, sid)) == 2


def test_rewritten_multimodal_replay_is_searchable_by_its_new_text(db):
    sid = "multimodal-rewrite"
    db.create_session(sid, "desktop")
    original = {"role": "assistant", "content": [{"type": "text", "text": "zebracornflake"}], "timestamp": 4000.0}
    assert db.append_messages_batch(sid, [original]) == 1

    restored = db.get_messages_as_conversation(sid, repair_alternation=True)
    restored[0]["content"] = [{"type": "text", "text": "quokkalantern"}]
    restored[0].pop(_DB_PERSISTED_MARKER, None)

    assert db.append_messages_batch(sid, restored) == 0

    assert db.search_messages("quokkalantern", limit=10)
    assert not db.search_messages("zebracornflake", limit=10)


def test_fts_backfill_between_replay_and_flush_keeps_the_live_edit(db):
    """The bounded FTS backfill fills historical rows' projection without touching content: it is not a
    foreign payload winner, so the replayed dict's edit must still be written."""
    sid = "backfill-race"
    db.create_session(sid, "desktop")
    original = {"role": "assistant", "content": [{"type": "text", "text": "before backfill"}], "timestamp": 5000.0}
    assert db.append_messages_batch(sid, [original]) == 1
    db._conn.execute("UPDATE messages SET fts_content = NULL WHERE session_id = ?", (sid,))
    db._conn.commit()

    restored = db.get_messages_as_conversation(sid, repair_alternation=True)
    db._backfill_fts_content(db._conn.cursor())
    db._conn.commit()
    restored[0]["content"] = [{"type": "text", "text": "after backfill"}]
    restored[0].pop(_DB_PERSISTED_MARKER, None)

    assert db.append_messages_batch(sid, restored) == 0
    assert db.get_messages_as_conversation(sid)[0]["content"] == [{"type": "text", "text": "after backfill"}]
