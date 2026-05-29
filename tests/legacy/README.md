# Legacy Test Archive

These files were the previous flat test suite. They are intentionally renamed
to `legacy_*.py` so pytest does not collect them by default.

Why they are archived:

- Several phase tests call real Tavily, ChromaDB embedding/reranking, or other
  heavyweight paths that are not safe for default CI.
- Some tests depend on a sibling frontend checkout rather than this backend
  repository.
- Some tests reflect older contracts that no longer match the current backend
  service code.
- Useful assertions were migrated into focused unit and integration tests under
  `tests/unit` and `tests/integration`.

Keep these files as historical reference while the new suite stabilizes. Do not
move them back into pytest discovery unless the external-service dependencies
are fully mocked or marked as non-default expensive tests.
