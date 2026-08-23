"""The exit codes the CLI promises, in one place.

```text
0  the command did what it was asked
1  something the caller can fix
```

Step 26.5's rule is that failure is communicated by the exit status rather than
by prose on stdout, so the codes are a contract and not an implementation detail
of whichever module happened to raise. Two are enough: a script asks "did this
work", and a caller who needs to know *what* went wrong reads the message on
stderr, which names the error rather than encoding it in a number.

They live in their own module because both the entry point and the commands that
decide their own status need them, and the entry point imports the commands.
"""

from __future__ import annotations

__all__ = ["EXIT_ERROR", "EXIT_OK"]

#: The command produced what it was asked for.
EXIT_OK = 0

#: Something the user can fix — an uninstalled provider, a failed build, a
#: config file that does not describe an executable fit. A traceback would say
#: the same thing less clearly and imply a bug.
EXIT_ERROR = 1
