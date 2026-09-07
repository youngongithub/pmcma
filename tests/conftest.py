"""Make scripts/ importable so tests can assert against the sample generator.

The correlation round-trip in test_desmooth.py needs the *same* parameters the
committed samples were built from. Importing them beats duplicating the magic
numbers, so a change to the generator breaks the test instead of silently
invalidating it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
