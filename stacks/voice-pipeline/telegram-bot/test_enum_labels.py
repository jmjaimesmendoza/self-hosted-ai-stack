"""Pure-python self-check for the enum glossary — no bot deps needed."""
import ast
from pathlib import Path

# Parse bot.py instead of importing it (imports need telegram/anthropic/etc.)
tree = ast.parse(Path(__file__).with_name("bot.py").read_text())
ns = {}
for node in tree.body:
    if isinstance(node, (ast.Assign, ast.FunctionDef)) and (
        getattr(node, "name", None) == "enum_labels_text"
        or any(getattr(t, "id", None) == "ENUM_GLOSSARY" for t in getattr(node, "targets", []))
    ):
        exec(compile(ast.Module([node], []), "bot.py", "exec"), ns)

ENUM_GLOSSARY, enum_labels_text = ns["ENUM_GLOSSARY"], ns["enum_labels_text"]

for col, vals in ENUM_GLOSSARY.items():
    for v, labels in vals.items():
        assert len(labels) == 3, f"{col}.{v} needs (ES, EN, PT), got {labels}"

assert "PREGNANT=Preñada" in enum_labels_text("ES")
assert "FEMALE=Hembra" in enum_labels_text("ES")
assert "PREGNANT=Pregnant" in enum_labels_text("EN")
assert "FEMALE=Fêmea" in enum_labels_text("PT")
assert enum_labels_text("XX") == enum_labels_text("EN")  # unknown locale -> EN

print("ok")
