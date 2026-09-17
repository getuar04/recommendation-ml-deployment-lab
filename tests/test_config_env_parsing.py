"""Targeted coverage for app.core.config's _getenv_int/_getenv_float helpers (P1 audit
fix): a malformed or empty EXPLICITLY-PROVIDED value must raise ValueError naming the
environment variable and the rejected raw value, never fall back to the default silently.
The absent-variable and valid-value paths are already exercised indirectly by every other
test in the suite (app.core.config is imported transitively via app.main on every test
run), so only the new failure-diagnostic behavior is covered here."""
import pytest

from app.core import config


@pytest.mark.parametrize("raw", ["", "abc", "12.5x", " "])
def test_getenv_int_malformed_value_raises_labeled_value_error(monkeypatch, raw):
    monkeypatch.setenv("SOME_TEST_INT_VAR", raw)
    with pytest.raises(ValueError, match=r"Invalid SOME_TEST_INT_VAR=.*expected integer"):
        config._getenv_int("SOME_TEST_INT_VAR", "10")


@pytest.mark.parametrize("raw", ["", "abc", "1,5"])
def test_getenv_float_malformed_value_raises_labeled_value_error(monkeypatch, raw):
    monkeypatch.setenv("SOME_TEST_FLOAT_VAR", raw)
    with pytest.raises(ValueError, match=r"Invalid SOME_TEST_FLOAT_VAR=.*expected float"):
        config._getenv_float("SOME_TEST_FLOAT_VAR", "0.5")
