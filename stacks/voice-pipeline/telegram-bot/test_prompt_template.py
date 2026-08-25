"""prompt.md must stay renderable by build_prompts — no bot deps needed.

The failure this catches: someone edits prompt.md, adds a literal `{` (a JSON
example, a psql format string) or drops a placeholder, and every query dies at
runtime with KeyError/IndexError deep inside build_prompts.
"""
import ast
import re
from pathlib import Path

HERE = Path(__file__).parent
template = (HERE / "prompt.md").read_text().rstrip("\n")
bot_src = (HERE / "bot.py").read_text()

# --- the keys build_prompts actually passes, read off the source
tree = ast.parse(bot_src)
build_prompts = next(n for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef) and n.name == "build_prompts")
fmt = next(n for n in ast.walk(build_prompts)
           if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "format")
supplied = {kw.arg for kw in fmt.keywords}
assert supplied == {"schema", "language", "enum_labels"}, supplied

# --- every brace in the file is a placeholder build_prompts supplies
used = set(re.findall(r"\{(\w*)\}", template))
assert used == supplied, f"prompt.md placeholders {used} != supplied {supplied}"
assert template.count("{") == template.count("}") == len(supplied), (
    "a stray brace in prompt.md will raise at render time — escape it as {{ or }}")

# --- it renders, and nothing is left unsubstituted
out = template.format(schema="SCHEMA_BLOCK", language="Spanish", enum_labels="ENUM_BLOCK")
for marker in ("SCHEMA_BLOCK", "ENUM_BLOCK", "Spanish"):
    assert marker in out, marker
assert not re.search(r"\{\w*\}", out), re.search(r"\{\w*\}", out).group()

# --- the parts that carry meaning survived the move out of bot.py
for phrase in ("DATABASE SCHEMA", "run_sql_query", "is_deleted", "EXAMPLE QUERIES",
               "organization_id", "NEVER invent"):
    assert phrase in out, f"prompt.md lost: {phrase}"

# --- and bot.py no longer carries a second copy of the prose
assert "FEW_SHOTS" not in bot_src and "GROUNDING" not in bot_src, \
    "prompt text still duplicated in bot.py"

print("ok")
