import sqlite3
from types import SimpleNamespace


def test_update_notice_reports_cjk_only_projection_debt(monkeypatch, capsys):
    from hermes_cli import config, update_cmd
    import hermes_constants
    import hermes_state

    monkeypatch.setattr(
        config, "load_config", lambda: {"sessions": {"fts_optimize_notice": "advise"}}
    )
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE messages_fts (content, tool_name, tool_calls);
        CREATE VIEW messages_fts_trigram_src AS
            SELECT 1 AS id, 'projected' AS fts_content;
        CREATE TABLE messages_fts_cjk (content, tool_name, tool_calls);
        CREATE VIEW messages_fts_cjk_src AS
            SELECT 1 AS id, 'legacy raw content' AS content;
        """
    )

    class FakePath:
        def __truediv__(self, _name):
            return self

        def exists(self):
            return True

        def stat(self):
            return SimpleNamespace(st_size=1024**3)

    class FakeSessionDB:
        def __init__(self, **_kwargs):
            self._conn = conn

        def close(self):
            pass

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: FakePath())
    monkeypatch.setattr(hermes_state, "SessionDB", FakeSessionDB)

    update_cmd._print_fts_optimize_available_notice()

    output = capsys.readouterr().out
    assert "Session search index upgrade available" in output
    assert "hermes sessions optimize-storage" in output
    conn.close()
