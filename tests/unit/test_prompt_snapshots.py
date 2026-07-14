"""
Snapshot tests proving the prompt-registry extraction is render-identical.

`capture_prompts()` invokes the LIVE prompt code (now registry-backed) with fixed
representative inputs and returns the exact strings sent to / returned by it. Each
is compared byte-for-byte against a checked-in snapshot that was generated from the
ORIGINAL inline strings before the registry rewire. A mismatch means a prompt's
rendered output changed — which must be accompanied by a version bump and a
deliberate snapshot update (see prompts/README.md).
"""

from __future__ import annotations

import pytest

from prompts import all_prompts, get_prompt
from prompts.base import PromptArtifact
from tests.unit.prompt_snapshot_cases import SNAPSHOT_DIR, capture_prompts

pytestmark = pytest.mark.unit


async def test_prompts_render_identical_to_snapshots():
    captured = await capture_prompts()
    assert captured, "capture_prompts returned nothing"
    mismatches = []
    for name, got in sorted(captured.items()):
        snapshot_file = SNAPSHOT_DIR / f"{name}.txt"
        assert snapshot_file.exists(), f"Missing snapshot file for '{name}'. Regenerate snapshots."
        expected = snapshot_file.read_text(encoding="utf-8")
        if got != expected:
            mismatches.append(name)
    assert not mismatches, (
        "Prompt render changed vs snapshot for: "
        + ", ".join(mismatches)
        + ". If intentional, bump the artifact version and update the snapshot."
    )


def test_every_registered_prompt_starts_at_version_1_or_higher():
    for name, art in all_prompts().items():
        assert isinstance(art, PromptArtifact)
        assert art.version >= 1, f"{name} has version < 1"
        assert art.template, f"{name} has an empty template"


def test_render_fails_loudly_on_missing_placeholder():
    seq = get_prompt("curriculum_sequencer_system")
    with pytest.raises(KeyError):
        seq.render()  # missing pace_hint


def test_render_fails_loudly_on_unexpected_kwarg():
    cov = get_prompt("curriculum_coverage_planner_system")
    with pytest.raises(KeyError):
        cov.render(nonexistent="x")  # static prompt takes no kwargs


def test_get_prompt_unknown_name_raises():
    with pytest.raises(KeyError):
        get_prompt("no_such_prompt")
