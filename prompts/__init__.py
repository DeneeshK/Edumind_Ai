"""
prompts/ — versioned prompt registry for EduMind's live agent flows.

Importing this package registers every PromptArtifact (one module per area). Use
``get_prompt(name)`` to fetch an artifact and ``artifact.render(**kwargs)`` to
produce the exact string sent to the model. See prompts/README.md for the
versioning rule and prompts/base.py for the primitives.
"""

from __future__ import annotations

from prompts.base import PromptArtifact, REGISTRY, all_prompts, get_prompt, register

# Import area modules for their registration side effects (order is irrelevant;
# names are globally unique and duplicate registration raises).
from prompts import curriculum, evaluation, lesson  # noqa: E402,F401

__all__ = [
    "PromptArtifact",
    "REGISTRY",
    "all_prompts",
    "get_prompt",
    "register",
]
