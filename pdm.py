"""
Backwards-compatible shim.

The implementation now lives in the ``pioneer_detection`` package
(pip install pioneer-detection). Importing ``from pdm import ...`` keeps
working for existing scripts, notebooks, and course material.

Reference:
    Vansteenberghe, Eric (2026),
    "Insurance supervision under climate change: a pioneer detection method,"
    The Geneva Papers on Risk and Insurance - Issues and Practice, 51(1), 176-207,
    https://doi.org/10.1057/s41288-025-00367-y
"""

from pioneer_detection.core import *  # noqa: F401,F403
from pioneer_detection.core import __all__  # noqa: F401
