"""Every canonical example, executed as a script.

Step 28's rule: an API change that breaks an example breaks CI. The docs suite
beside this one executes the blocks on a page; this executes the files an agent
is told to run, in the form it is told to run them — a subprocess, one script,
no test harness in the namespace. Anything an example only works because pytest
imported it fails here.

Each run gets a home directory and a working directory of its own. The default
artifact store lives under the user data directory, so a suite that fitted models
into the real one would be a suite that changed the machine it ran on — and a
working directory in ``tmp_path`` is what proves an example bundles its data
rather than reading a file that happens to sit beside the repository root.

That isolation has one consequence worth naming: an installed provider
environment lives under the same user data directory, so no example ever sees one
here. ``05_probabilistic.py`` and ``07_zero_shot.py`` read the catalog and take
their fallback branch, which is the branch this suite checks — the branch that
needs ``openforecast providers install`` is checked by the integration suites
under ``integrations/``, the way every other provider-dependent claim in this
repository is.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples"
INDEXES = (
    EXAMPLES_ROOT / "README.md",
    REPO_ROOT / "docs" / "getting-started" / "examples.md",
)

#: Step 28.2: an agent that has cloned the repository can run these. Anything
#: that reads the network or a data file, or that draws a random number, is an
#: example whose output depends on something it did not bring with it. Matched
#: against what the code *does* rather than against the prose around it — the
#: word "requests" belongs in a sentence about request objects.
FORBIDDEN = (
    (r"^\s*(?:import|from)\s+(?:random|secrets)\b", "a random number makes two runs disagree"),
    (r"^\s*(?:import|from)\s+(?:urllib|http|socket)\b", "an example may not reach the network"),
    (r"^\s*(?:import|from)\s+(?:requests|httpx|aiohttp)\b", "an example may not reach the network"),
    (r"\bread_(?:csv|parquet|json|table)\s*\(", "an example bundles or generates its data"),
    (r"\bdatetime\.now\s*\(|\butcnow\s*\(", "a wall clock makes two runs disagree"),
)

_NUMBERED = re.compile(r"^(?P<number>\d\d)_[a-z0-9_]+\.py$")


def example_scripts() -> tuple[Path, ...]:
    return tuple(path for path in sorted(EXAMPLES_ROOT.glob("*.py")) if _NUMBERED.match(path.name))


EXAMPLES = example_scripts()


def run(script: Path, home: Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Execute ``script`` the way the documentation says to, in isolation."""
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(home),
        "USERPROFILE": str(home),
        "XDG_DATA_HOME": str(home / "data"),
        "XDG_CACHE_HOME": str(home / "cache"),
        "XDG_STATE_HOME": str(home / "state"),
        # The examples print rows; a narrow terminal would not change what they
        # do, but it does change what a failure message shows.
        "COLUMNS": "120",
    }
    return subprocess.run(  # noqa: S603 - the interpreter running this suite
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=environment,
        timeout=600,
        check=False,
    )


@pytest.fixture
def isolated_home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    return home


def test_there_are_examples() -> None:
    """Otherwise a suite that found none would pass by finding nothing to do."""
    assert EXAMPLES, f"no numbered examples under {EXAMPLES_ROOT}"


def test_the_examples_are_numbered_from_one_without_gaps() -> None:
    """The numbers are a reading order, so a gap in them is a missing lesson."""
    numbers = [
        int(match.group("number")) for path in EXAMPLES if (match := _NUMBERED.match(path.name))
    ]

    assert numbers == list(range(1, len(numbers) + 1)), f"numbering has a gap: {numbers}"


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda path: path.name)
def test_the_example_runs_as_a_script(script: Path, isolated_home: Path, tmp_path: Path) -> None:
    """Step 28.3: `uv run examples/03_point_in_time.py` works, or CI fails."""
    completed = run(script, isolated_home, tmp_path)

    assert completed.returncode == 0, (
        f"{script.relative_to(REPO_ROOT)} exited {completed.returncode}\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )
    assert completed.stdout.strip(), (
        f"{script.relative_to(REPO_ROOT)} printed nothing; an example a reader "
        f"cannot see the result of is a test rather than an example"
    )


def test_a_broken_example_would_fail(isolated_home: Path, tmp_path: Path) -> None:
    """The check above reports through pytest, so this asserts the mechanism.

    A runner that reported success whatever the script did would make every
    assertion above vacuous.
    """
    broken = tmp_path / "99_broken.py"
    broken.write_text("raise ValueError('a stale example')\n", encoding="utf-8")

    completed = run(broken, isolated_home, tmp_path)

    assert completed.returncode != 0
    assert "a stale example" in completed.stderr


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda path: path.name)
def test_the_example_says_how_to_run_itself(script: Path) -> None:
    """The first thing a reader needs is the command, and it names this file."""
    docstring = script.read_text(encoding="utf-8")

    assert f"uv run examples/{script.name}" in docstring, (
        f"{script.name} does not tell a reader how to run it"
    )


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda path: path.name)
def test_the_example_brings_its_own_data(script: Path) -> None:
    """Step 28.2: tiny, deterministic, and generated by the example itself."""
    source = script.read_text(encoding="utf-8")
    found = [
        reason
        for pattern, reason in FORBIDDEN
        if re.search(pattern, source, flags=re.MULTILINE) is not None
    ]

    assert not found, f"{script.name}: {found}"


def test_the_data_check_would_notice_an_example_that_reached_out() -> None:
    """A pattern set that matched nothing would make the check above vacuous."""
    offending = (
        "import random\n"
        "import requests\n"
        "frame = pandas.read_csv('prices.csv')\n"
        "now = datetime.now()\n"
    )
    caught = {
        reason
        for pattern, reason in FORBIDDEN
        if re.search(pattern, offending, flags=re.MULTILINE) is not None
    }

    assert caught == {reason for _, reason in FORBIDDEN}


@pytest.mark.parametrize("index", INDEXES, ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_every_example_is_listed(index: Path) -> None:
    """Step 28.4: one list per audience, and neither may go stale.

    ``examples/README.md`` is what a reader of the repository finds and the
    documentation page is what a reader of the site finds. Both are hand-written
    prose *about* the scripts rather than a second copy of them, which is exactly
    the kind of thing that rots — so an example added without a line here, or a
    line left behind by an example that was renamed, is a failure.
    """
    listed = set(re.findall(r"\d\d_[a-z0-9_]+\.py", index.read_text(encoding="utf-8")))
    present = {script.name for script in EXAMPLES}

    assert listed == present, (
        f"{index.relative_to(REPO_ROOT)} and examples/ disagree:\n"
        f"  listed, not present: {sorted(listed - present)}\n"
        f"  present, not listed: {sorted(present - listed)}"
    )
