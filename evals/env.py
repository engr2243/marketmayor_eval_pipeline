"""
The few facts about the environment the pipeline needs.

Small on purpose. In the private version this module is considerably larger,
because there the application is not an installed package and has to be put on
`sys.path` before anything can import it. Here the demo application sits beside
the pipeline in the same project, so there is nothing to bootstrap — which is
worth knowing if you are reading this as a template: the awkward part of
evaluating a real system is usually getting at it, not scoring it.

Nothing in this module reads, logs or prints an API key.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def api_key() -> str:
    """
    The key for eval runs, from the environment and nowhere else.

    `EVAL_ANTHROPIC_API_KEY` first, so eval spend can be billed to a separate
    key from whatever the application uses. Never logged, never written to a run
    record, never echoed in a report.
    """
    key = os.environ.get("EVAL_ANTHROPIC_API_KEY") or os.environ.get(
        "ANTHROPIC_API_KEY", ""
    )
    return key.strip()


def git_sha() -> str:
    """The commit a run was made at, for reproducibility. Empty if unknown."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""
