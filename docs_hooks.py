"""What the built site needs that a static-site generator does not produce.

`mkdocs.yml` names this file under `hooks:`, so it runs as part of an ordinary
`mkdocs build` with no plugin to install. Everything it does is Step 29's: the
whole corpus in one file, and the Markdown *source* of every page served beside
its rendered HTML, so a reader with a fetch tool never has to parse a themed
page to read a sentence.

The work itself lives in `openforecast.docs.llms`, where the tests can reach it
without a static-site generator installed. This file is the seam.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openforecast.docs import llms


def on_post_build(config: Any, **_kwargs: Any) -> None:
    """After the HTML: `llms-full.txt`, and one `.md` per page."""
    for path in llms.write_site(Path(config["site_dir"])):
        print(f"agent-readable: {path}")
