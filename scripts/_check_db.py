import sqlite3, json
from pathlib import Path

BOB_DB = Path.home() / ".bob" / "db" / "bob.db"
with sqlite3.connect(f"file:{BOB_DB}?mode=ro", uri=True) as db:
    rows = db.execute(
        "SELECT project_id, approval_config FROM tasks"
        " WHERE task_type = 'normal' ORDER BY updated_at DESC LIMIT 5"
    ).fetchall()
for pid, cfg in rows:
    print(repr(pid), json.loads(cfg or "{}"))
