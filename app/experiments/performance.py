"""Session 5 (spec §60): low-overhead performance measurement. Every duration uses
`time.perf_counter()`, is reported in seconds (documented unit everywhere this feeds into a
report), is never hard-coded, and never influences ranking/output content -- this is a pure
side-channel measurement wrapped around the exact functions the app already calls.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def measure() -> Iterator[dict]:
    """`with measure() as timing: ...` -- `timing["seconds"]` is populated (rounded to
    microsecond precision) once the block exits, including on exception."""
    result: dict = {"seconds": None}
    started = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = round(time.perf_counter() - started, 6)
