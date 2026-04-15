"""Helpers for using the vendored ABIDES source tree."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_abides_paths() -> Path:
    """Add the vendored ABIDES packages to ``sys.path`` and return the repo root."""

    project_root = Path(__file__).resolve().parent
    abides_root = project_root / "external" / "abides-jpmc-public"
    if not abides_root.exists():
        raise FileNotFoundError(
            "Vendored ABIDES repository not found at external/abides-jpmc-public."
        )

    package_roots = [
        abides_root / "abides-core",
        abides_root / "abides-markets",
    ]
    for package_root in reversed(package_roots):
        package_root_str = str(package_root)
        if package_root_str not in sys.path:
            sys.path.insert(0, package_root_str)

    return abides_root
