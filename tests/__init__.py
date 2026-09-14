"""Test suite package initialization with pytest fallback shim for python3 -m unittest."""

import math
import sys

try:
    import pytest
except ImportError:
    class _Approx:
        def __init__(self, expected, rel=None, abs=None):
            self.expected = expected
            self.rel = rel
            self.abs = abs

        def __eq__(self, actual):
            if self.abs is not None:
                return math.isclose(actual, self.expected, abs_tol=self.abs)
            return math.isclose(actual, self.expected, rel_tol=self.rel or 1e-4)

        def __repr__(self):
            return f"approx({self.expected})"

    class _PytestShim:
        approx = _Approx

    sys.modules["pytest"] = _PytestShim()
