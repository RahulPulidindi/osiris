"""Osiris: an autonomous trading agent for a Robinhood Agentic account.

The dependency check below runs before any submodule imports. Without it, running
the package with the wrong interpreter produces a bare
`ModuleNotFoundError: No module named 'pydantic'` and a traceback pointing at
`config.py` -- which reads like a broken install rather than what it actually is:
the system Python being used instead of the virtualenv. That is the single most
likely first-run mistake, and the default message sends you to the wrong file.
"""

from __future__ import annotations

__version__ = "0.1.0"


def _check_environment() -> None:
    import importlib.util
    import sys
    from pathlib import Path

    # `pydantic` stands in for the whole dependency set: it is imported by
    # `config`, which nearly everything else imports.
    if importlib.util.find_spec("pydantic") is not None:
        return

    # src/osiris/__init__.py -> src -> repo root
    repo_root = Path(__file__).resolve().parent.parent.parent
    venv_python = repo_root / ".venv" / "bin" / "python"
    lines = [
        "",
        "Osiris dependencies are not available to this interpreter.",
        f"  interpreter: {sys.executable}",
    ]
    if venv_python.exists():
        # The common case: the venv exists and simply was not activated.
        lines += [
            "",
            "A virtualenv exists but is not active. Either activate it:",
            "",
            f"    source {repo_root / '.venv' / 'bin' / 'activate'}",
            "",
            "or call its interpreter directly:",
            "",
            f"    {venv_python} -m osiris.connect",
        ]
    else:
        lines += [
            "",
            "No virtualenv found. Create one and install the package:",
            "",
            f"    cd {repo_root}",
            "    python3 -m venv .venv",
            "    source .venv/bin/activate",
            '    pip install -e ".[dev]"',
        ]
    lines.append("")
    raise SystemExit("\n".join(lines))


_check_environment()
del _check_environment
