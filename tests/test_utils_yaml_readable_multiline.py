"""Preserve the fork's readable multiline YAML contract."""

import yaml

from utils import atomic_yaml_write


def test_atomic_yaml_write_uses_block_scalars_for_multiline_text(tmp_path):
    target = tmp_path / "config.yaml"
    prompt = "一行目\n- 二行目\n三行目\n"
    data = {"slack": {"channel_prompts": {"C123": prompt}}}

    atomic_yaml_write(target, data)

    raw = target.read_text(encoding="utf-8")
    assert "C123: |" in raw
    assert "\\n" not in raw
    assert yaml.safe_load(raw) == data
