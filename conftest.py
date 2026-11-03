"""
Make `evals` and `demo_app` importable however pytest was invoked.

The pipeline and the demo application are top-level packages at the repository
root and this is not an installed project, so the root has to be on `sys.path`
before either import resolves. pytest's own rootdir insertion depends on
invocation style; this does not.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
