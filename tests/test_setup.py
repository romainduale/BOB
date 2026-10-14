"""Check that every step of docs/SETUP.md is done on this machine.

Run from the repo root, with the virtual environment active:

    python -m pytest tests/test_setup.py -v

or, equivalently:

    python tests/test_setup.py

Each test matches one step of SETUP.md. A failing test tells you which step
to redo.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SKORE_FILE = REPO_ROOT / ".skore"
SKILLS_DIR = REPO_ROOT / ".bob" / "skills"
BOB_DB = Path.home() / ".bob" / "db" / "bob.db"
HUB_URL = "https://api.skore.probabl.ai"
SKILLS_REPO = "probabl-ai/skills-hackathon"
INSTALL_SKILLS = f"skore skills install all --repo {SKILLS_REPO} --agent bob"


def setup_fail(message: str) -> None:
    """Fail a setup check with a plain, wrapped instruction.

    Re-flow the text to a readable width and pad it with blank lines so the
    failure tells the participant what to do instead of showing a raw assert
    expression clamped to one line.
    """
    body = textwrap.fill(" ".join(message.split()), width=72)
    pytest.fail(f"\n{textwrap.indent(body, '  ')}\n", pytrace=False)


def load_skore_file() -> dict:
    if not SKORE_FILE.is_file():
        setup_fail(
            ".skore is missing at the repo root. "
            "Run `python scripts/skore-agent` (SETUP.md step 5)."
        )
    try:
        return json.loads(SKORE_FILE.read_text())
    except json.JSONDecodeError:
        setup_fail(
            ".skore is not valid JSON. Delete it and run "
            "`python scripts/skore-agent` again."
        )


# Step 2: Python


def test_python_version_is_3_12_or_newer():
    if sys.version_info < (3, 12):
        setup_fail(
            f"Python {sys.version.split()[0]} is too old. "
            "Install Python 3.12 or newer (SETUP.md step 2)."
        )


def test_virtual_environment_is_active():
    expected = (REPO_ROOT / ".venv").resolve()
    if Path(sys.prefix).resolve() != expected:
        setup_fail(
            f"Tests are running with {sys.executable}, not the repo's .venv. "
            "Activate it first with `source .venv/bin/activate` (macOS/Linux) "
            "or `.venv\\Scripts\\Activate.ps1` (Windows)."
        )


# Step 4: skore install


def test_skore_cli_is_installed():
    try:
        import skore_cli  # noqa: F401
    except ImportError:
        setup_fail(
            "skore-cli is not installed in this environment. "
            "Run `python -m pip install -r requirements.txt`."
        )


EXPECTED_SKILLS = [
    "audit-ml-pipeline",
    "build-ml-pipeline",
    "data-science-python-stack",
    "evaluate-ml-pipeline",
    "explore-ml-data",
    "iterate-from-skore",
    "iterate-from-user",
    "iterate-ml-experiment",
    "organize-ml-workspace",
    "python-api",
    "python-code-style",
    "python-env-manager",
    "smoke-test-ml-pipeline",
    "test-ml-pipeline",
]


def test_bob_skills_are_installed():
    catalog_path = SKILLS_DIR / ".catalog.json"
    if not catalog_path.is_file():
        setup_fail(f"No lab skills found in {SKILLS_DIR}. Run `{INSTALL_SKILLS}`.")

    catalog = json.loads(catalog_path.read_text())
    source = catalog.get("sources", {}).get(SKILLS_REPO)
    if not source:
        setup_fail(
            f"The skills in .bob/skills do not come from {SKILLS_REPO}. "
            f"Run `{INSTALL_SKILLS}`."
        )

    missing = [
        name for name in EXPECTED_SKILLS if not (SKILLS_DIR / name / "SKILL.md").is_file()
    ]
    if missing:
        setup_fail(
            f"These lab skills are missing from .bob/skills: {', '.join(missing)}. "
            f"Run `{INSTALL_SKILLS}` again."
        )

    installed = sorted(
        p.name for p in SKILLS_DIR.iterdir()
        if p.is_dir() and (p / "SKILL.md").is_file()
    )
    if installed != EXPECTED_SKILLS:
        setup_fail(
            "Installed skills do not match the expected set. "
            f"Expected: {', '.join(EXPECTED_SKILLS)}. "
            f"Installed: {', '.join(installed)}. "
            f"Run `{INSTALL_SKILLS}` to restore the correct skill set."
        )


def test_skore_file_is_complete():
    config = load_skore_file()

    missing = [
        key
        for key in ("hub_url", "workspace", "workspace_id", "api_key")
        if config.get(key) in (None, "")
    ]
    if missing:
        setup_fail(
            f".skore is missing {', '.join(missing)}. Delete it and run "
            "`python scripts/skore-agent` again."
        )
    if config["hub_url"].rstrip("/") != HUB_URL:
        setup_fail(
            f".skore points to {config['hub_url']}, not the lab hub {HUB_URL}. "
            "Delete it and run `python scripts/skore-agent` again."
        )


# Step 3 and 4: Skore Hub account, team workspace, sign-in


def test_hub_accepts_the_api_key():
    config = load_skore_file()
    workspace = urllib.parse.quote(config["workspace"], safe="")
    url = f"{HUB_URL}/projects/{workspace}/bobathon-esilv/estimator-reports/"
    request = urllib.request.Request(url, headers={"X-API-Key": config["api_key"]})

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    except urllib.error.URLError as exc:
        setup_fail(f"Could not reach {HUB_URL}: {exc.reason}. Check your internet.")

    if status in (401, 403):
        setup_fail(
            f"Skore Hub rejected the API key in .skore (HTTP {status}). "
            "Make sure you joined your team workspace on https://skore.probabl.ai, "
            "then delete .skore and run `python scripts/skore-agent` again."
        )
    # 404 is fine: the `bobathon-esilv` project is created by your first `put` on Day 1.
    if status not in (200, 404):
        setup_fail(f"Unexpected answer from Skore Hub: HTTP {status}.")


# Bob Permissions checks


def _same_workspace(project_id: str) -> bool:
    path = project_id.removeprefix("file:").replace("\\", "/").lstrip("/").lower()
    return path == REPO_ROOT.as_posix().lstrip("/").lower()


def test_bob_skill_permission_is_toggled():
    """Check that Skill is ticked in Bob's Permissions menu for this project.

    Bob saves the toggles on each chat task in ~/.bob/db/bob.db; this reads the
    most recently used chat opened on this repo.
    """
    open_chat = (
        "Open this repo folder in Bob IDE, start a chat, then click Permissions "
        "(next to the mode selector at the bottom of the chat) and tick Skill."
    )
    if not BOB_DB.is_file():
        setup_fail(f"Bob has never run on this machine. {open_chat}")

    with sqlite3.connect(f"{BOB_DB.as_uri()}?mode=ro", uri=True) as db:
        rows = db.execute(
            "SELECT project_id, approval_config FROM tasks "
            "WHERE task_type = 'normal' ORDER BY updated_at DESC"
        ).fetchall()

    configs = [config for project_id, config in rows if _same_workspace(project_id)]
    if not configs:
        setup_fail(f"No Bob chat found for {REPO_ROOT}. {open_chat}")

    config = json.loads(configs[0] or "{}")
    allowed = config.get("allowed_permissions", [])
    if not config.get("autoApprovalEnabled") or "skill" not in allowed:
        setup_fail(
            "Skill is not ticked in Bob's Permissions menu. Open the Bob chat "
            "for this repo, click Permissions (next to the mode selector at the "
            "bottom of the chat), tick Skill, keep Auto-approve turned on "
            "(SETUP.md step 4), then re-run the tests."
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))