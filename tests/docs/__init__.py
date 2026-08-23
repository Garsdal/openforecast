"""Step 25: the documentation, checked the way the code is.

Two things are asserted here, and both of them are what stops documentation drift
from being a discovery rather than a diff:

- every Python example in ``docs/`` executes, page by page, in the order a reader
  meets it (:mod:`tests.docs.test_docs_examples`)
- the site's structure holds: the nav names every page and every page is in the
  nav, and every relative link resolves to a file that exists
  (:mod:`tests.docs.test_docs_structure`)

The generated reference pages are checked in ``tests/unit/test_reference_docs.py``
instead, beside the generator that writes them.
"""
