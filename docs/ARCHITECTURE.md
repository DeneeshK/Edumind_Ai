# Architecture

EduMind's architecture doc is versioned, the same way the code it describes
is — so "how does this work" always has an answer for a specific point in
time, not just whatever's true today.

| Version | Status | Covers |
| --- | --- | --- |
| [V2](architecture/ARCHITECTURE_V2.md) | **Current — read this** | Live vs legacy code paths, request flows, data model, prompt registry + golden evals, guardrails, observability (tracing + token/cost). |
| [V1](architecture/ARCHITECTURE_V1.md) | Archived | The system before retrieval cleanup, tracing, the prompt registry, and guardrails existed. Kept for diffing against V2. |

Start with [ARCHITECTURE_V2.md](architecture/ARCHITECTURE_V2.md) — it's the
one that matches the deployed system. V2's own
["Evolution from V1"](architecture/ARCHITECTURE_V2.md#evolution-from-v1)
section summarizes what changed and why, if you want the short version before
reading either doc in full.
