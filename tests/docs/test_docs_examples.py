"""Every Python example in the documentation, executed.

Step 25's "done when" is that documentation drift becomes difficult to introduce
accidentally. A generated reference makes a stale *signature* a diff; this makes a
stale *example* a failure. If an API example no longer executes, CI fails.

A page is executed as a page: its blocks run in order, in one namespace, so the
second example may use the frame the first one built — which is also the order a
reader meets them in. Blocks that cannot run here carry a skip marker naming the
reason, and the marker is checked as strictly as the code: a reason is required,
and a page where nothing at all runs has to be one where nothing *can*.

The examples fit models, so each page gets its own home directory: the default
artifact store is under the user data directory, and a test suite that wrote to
the real one would be a test suite that changed the machine it ran on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.docs.blocks import REPO_ROOT, Block, blocks_in, pages_under

#: The hand-written pages, plus the README: a Python example in the README is as
#: public as one in a guide, and the same rule applies to it — it runs here, or it
#: says in the source of the page why it cannot.
PAGES = (*pages_under(), REPO_ROOT / "README.md")

#: Pages that are entirely prose, or entirely about code that needs a provider
#: environment. Listed rather than inferred, so that a page which *stops*
#: executing anything has to be added here deliberately.
NOTHING_TO_EXECUTE = {
    Path("docs/index.md"),
    # The README's one example is the cross-provider, cross-lifecycle backtest,
    # which needs three provider environments. The executable version of it is
    # `examples/06_ensemble.py`, against the built-in model.
    Path("README.md"),
    # Step 28: the executable examples are the scripts under `examples/`, run as
    # scripts by `tests/examples/test_examples.py`. This page is the index to
    # them, and copying one onto it is the duplication 28.4 is about.
    Path("docs/getting-started/examples.md"),
    Path("docs/concepts/data-model.md"),
    Path("docs/concepts/point-in-time.md"),
    Path("docs/concepts/providers.md"),
    Path("docs/integrations/nixtla.md"),
    Path("docs/integrations/darts.md"),
    Path("docs/integrations/sktime.md"),
    Path("docs/integrations/sklearn.md"),
    Path("docs/integrations/chronos.md"),
}


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A home directory of this test's own, and a working directory to match."""
    home = tmp_path / "home"
    home.mkdir()
    for variable in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(variable, str(home))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / "cache"))
    monkeypatch.setenv("XDG_STATE_HOME", str(home / "state"))
    monkeypatch.chdir(tmp_path)
    return home


@pytest.mark.parametrize("page", PAGES, ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_every_example_on_the_page_runs(page: Path, isolated_home: Path) -> None:
    """One namespace per page, blocks in the order they are written."""
    namespace: dict[str, Any] = {"__name__": "__docs__"}
    executed = 0
    for block in blocks_in(page):
        if not block.is_python or block.skip_reason is not None:
            continue
        code = compile(block.source, block.location, "exec")
        try:
            exec(code, namespace)  # noqa: S102 - executing the documentation is the point
        except Exception as error:  # pragma: no cover - the failure is the message
            pytest.fail(f"{block.location} raised {type(error).__name__}: {error}")
        executed += 1

    relative = page.relative_to(REPO_ROOT)
    if relative in NOTHING_TO_EXECUTE:
        assert executed == 0, f"{relative} now executes {executed} blocks; drop it from the list"
    else:
        assert executed, f"{relative} executes nothing; add it to NOTHING_TO_EXECUTE with a reason"


def test_every_skipped_example_says_why() -> None:
    """A block that is not executed is a claim, and a claim needs a reason."""
    unexplained = [
        block.location
        for page in PAGES
        for block in blocks_in(page)
        if block.is_python and block.skip_reason is not None and len(block.skip_reason) < 10
    ]
    assert not unexplained, f"skip markers without a real reason: {unexplained}"


def test_most_python_examples_are_executed() -> None:
    """The skip marker is an exception, and this is what keeps it one.

    A suite that skipped everything would pass; the ratio is what says the docs
    are still being run rather than being described as unrunnable.
    """
    python = [block for page in PAGES for block in blocks_in(page) if block.is_python]
    executed = [block for block in python if block.skip_reason is None]

    assert len(executed) >= len(python) // 2, (
        f"only {len(executed)} of {len(python)} Python examples are executed"
    )


def test_the_block_reader_finds_markers_and_languages(tmp_path: Path) -> None:
    """Otherwise a reader that found nothing would look like a suite that passed."""
    page = tmp_path / "example.md"
    page.write_text(
        "# Title\n\n"
        "```python\n"
        "answer = 42\n"
        "```\n\n"
        "<!-- docs-exec: skip — needs a provider environment -->\n\n"
        "```python\n"
        "of.fit(model='nixtla/nhits', data=data)\n"
        "```\n\n"
        "```bash\n"
        "openforecast serve\n"
        "```\n",
        encoding="utf-8",
    )

    found = blocks_in(page)

    assert [block.language for block in found] == ["python", "python", "bash"]
    assert [block.skip_reason for block in found] == [
        None,
        "needs a provider environment",
        None,
    ]
    assert found[0].source == "answer = 42\n"


def test_the_examples_are_executed_rather_than_only_parsed(isolated_home: Path) -> None:
    """The harness runs code, and a broken example really does fail.

    The check above reports a failure as a pytest failure, so this asserts the
    mechanism underneath it on a block that cannot possibly work.
    """
    block = Block(
        path=REPO_ROOT / "docs" / "example.md",
        line=1,
        language="python",
        source="raise ValueError('a stale example')\n",
        skip_reason=None,
    )

    with pytest.raises(ValueError, match="a stale example"):
        exec(compile(block.source, block.location, "exec"), {})  # noqa: S102
