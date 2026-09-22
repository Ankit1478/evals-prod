"""Integration shim for the vendored LLM_AS_JUDGE suite.

Upstream, pytest runs from that project's own root, so a handful of its
tests load fixtures by working-directory-relative path ("datasets/...").
Here the suite runs from the Evals-prod root instead.

This restores the working directory those tests expect, which is why every
file under vendor/llm_judge/ can stay byte-identical to upstream - a
re-sync stays a plain rsync, with this one added file as the only seam.
monkeypatch.chdir puts the working directory back after each test.
"""

from pathlib import Path

import pytest

VENDOR_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _run_from_vendor_root(monkeypatch):
    monkeypatch.chdir(VENDOR_ROOT)
