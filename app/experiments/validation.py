"""One canonical identifier validator, reused by configuration loading (app.core.config)
and by report persistence (app.experiments.report_store) so the two never drift apart.

Deliberately dependency-free (no app-internal imports) to avoid import cycles with
app.core.config, which is imported by nearly everything else in the app.
"""
from __future__ import annotations

import re

# Conservative allowlist: letters/digits/._- only would still permit ".." as a standalone
# path-traversal segment, so "." is excluded entirely -- there is never a legitimate reason
# for a dot in an experiment/dataset/volume identifier. 1-64 characters, must start with a
# letter or digit (rules out a leading "-" being mistaken for a CLI flag, and rules out an
# empty match). This rejects path separators, "..", control characters, blank values, and
# unreasonable length by construction (whitelist, not a blacklist of "bad" characters).
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class InvalidIdentifierError(ValueError):
    pass


def validate_identifier(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.match(value):
        raise InvalidIdentifierError(
            f"{field} must be 1-64 characters of letters, digits, '_' or '-', starting with "
            f"a letter or digit (got {value!r})"
        )
    return value
