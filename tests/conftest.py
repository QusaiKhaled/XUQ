"""
Pytest configuration: ensure project root is on sys.path before any test module is loaded.
This makes 'src' importable when running pytest from the repo root (e.g. in CI).
"""
import sys
from pathlib import Path

# Project root = directory that contains the 'tests' folder
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
