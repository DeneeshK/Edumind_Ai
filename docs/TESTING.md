# Testing

The project uses pytest with async support. Test configuration lives in
`pytest.ini`.

## Default Test Command

```bash
venv/bin/pytest -q
```

`pytest.ini` collects:

- `tests/unit`
- `tests/integration`

Legacy scripts in `tests/legacy` are intentionally outside the default
collection path.

## Test Groups

| Path | Purpose |
| --- | --- |
| `tests/unit/` | Fast service, config, prompt, schema, model, report-writer, and client behavior tests. |
| `tests/integration/` | Lightweight API and course-flow tests with mocked external dependencies. |
| `tests/fixtures/` | Sample course, roadmap, and student-state payloads. |
| `tests/legacy/` | Historical tests kept for reference, not part of the default suite. |

## Useful Commands

Run everything:

```bash
venv/bin/pytest -q
```

Run only unit tests:

```bash
venv/bin/pytest tests/unit -q
```

Run only integration tests:

```bash
venv/bin/pytest tests/integration -q
```

Run a single file:

```bash
venv/bin/pytest tests/unit/test_course_service.py -q
```

Run a single test:

```bash
venv/bin/pytest tests/unit/test_course_service.py::test_generate_module_lesson_saves_mocked_content_without_questions -q
```

## External Services in Tests

The current default suite should not require real Groq, Google, Tavily, or
PostgreSQL network calls. Tests use monkeypatching, fixtures, and FastAPI test
clients for mocked behavior.

When adding tests around LLM-backed flows:

- Mock `clients.groq_client.generate`, `stream`, or `tool_call_loop`.
- Assert saved payloads and service outputs instead of full generated prose.
- Avoid snapshotting long lesson markdown unless the content is deterministic.
- Keep secrets out of fixtures.

## What To Test For Course Changes

Course creation and module generation changes should usually cover:

- `tests/integration/test_course_creation_mocked.py`
- `tests/integration/test_module_generation_mocked.py`
- `tests/integration/test_roadmap_generation_mocked.py`
- `tests/unit/test_course_service.py`
- `tests/unit/test_roadmap_service.py`
- `tests/unit/test_prompt_builders.py`

Auth changes should usually cover:

- `tests/unit/test_auth_utils.py`
- `tests/integration/test_auth_flow_mocked.py`

Evaluation/report changes should usually cover:

- `tests/unit/test_evaluation_report_writer.py`
- `tests/unit/test_offline_metrics_runner.py`
- `tests/unit/test_llm_client_mocked.py`

## Documentation-Only Change Checks

For docstring/comment-only Python changes, use both pytest and an AST comparison
that strips docstrings before comparing with `HEAD`. This confirms executable
Python structure did not change.

```bash
python3 - <<'PY'
import ast, subprocess
from pathlib import Path

def strip_docstrings(node):
    if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            node.body = body[1:]
    for child in ast.iter_child_nodes(node):
        strip_docstrings(child)
    return node

changed = subprocess.check_output(["git", "diff", "--name-only"], text=True).splitlines()
py_files = [p for p in changed if p.endswith(".py")]
failures = []
for path in py_files:
    current = Path(path).read_text(encoding="utf-8")
    original = subprocess.check_output(["git", "show", f"HEAD:{path}"], text=True)
    cur_tree = strip_docstrings(ast.parse(current))
    orig_tree = strip_docstrings(ast.parse(original))
    if ast.dump(cur_tree, include_attributes=False) != ast.dump(orig_tree, include_attributes=False):
        failures.append(path)
if failures:
    print("\n".join(failures))
    raise SystemExit(1)
print(f"AST unchanged after stripping docstrings for {len(py_files)} changed Python files")
PY
```

Also run:

```bash
git diff --check
venv/bin/pytest -q
```
