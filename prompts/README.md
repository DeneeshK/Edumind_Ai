# Prompt registry (`prompts/`)

Versioned, tracked prompt artifacts for EduMind's **live** agent flows. Each prompt
is a `PromptArtifact` (name, integer version, template string, `render(**kwargs)`)
registered in a global `REGISTRY`. Fetch one with `get_prompt(name)`.

```python
from prompts import get_prompt

system = get_prompt("curriculum_sequencer_system").render(pace_hint=hint_text)
```

## Layout

| Module | Prompts |
| --- | --- |
| `prompts/curriculum.py` | coverage-planner system, sequencer system + pace rules, auditor system |
| `prompts/evaluation.py` | evaluation system, diagnose / probe / finalize instruction blocks |
| `prompts/lesson.py` | lesson-generation scaffold + pace blocks, question-generation retry, module-chat prompts |

Placeholders use `{{name}}` (double braces) so they never collide with the single
braces of the JSON schemas embedded in the prompts. `render()` **fails loudly** if a
placeholder is left unfilled or an unexpected kwarg is passed — a typo cannot
silently alter a live prompt. The call sites assemble the dynamic data (JSON metadata
blocks, concept lists) and pass it in; the authored prompt text lives here.

## Versioning rule (read before editing a prompt)

1. **Bump `version`** (integer, +1) on ANY semantic edit to a template — reworded
   instructions, changed rules, added/removed guidance. Whitespace-only reflow still
   counts if it changes the rendered string.
2. **Update the snapshot in the SAME commit.** Every prompt is snapshot-tested in
   `tests/unit/test_prompt_snapshots.py` against files in `tests/unit/snapshots/`.
   Regenerate with:
   ```bash
   python -m pytest tests/unit/test_prompt_snapshots.py   # will fail, showing the diff
   # then, intentionally, regenerate the snapshot fixtures and re-run
   ```
   (A helper capture lives in `tests/unit/prompt_snapshot_cases.py`.)
3. New prompts start at **version 1**.

## Traceability

When a registry prompt drives a `clients.groq_client.generate` call, the call passes
`_prompt_name` / `_prompt_version`. These are attached to the LLM span as
`gen_ai.prompt.name` / `gen_ai.prompt.version` (OpenTelemetry, opt-in). The params are
optional and default to `None` — zero behaviour change when absent.

## Golden evals

The golden regression suite (`evaluation/golden/`) exercises the live flows that use
these prompts and fails CI when a prompt edit degrades curriculum quality or answer
diagnosis. See `docs/ARCHITECTURE.md` → "Prompt management and golden evals".
