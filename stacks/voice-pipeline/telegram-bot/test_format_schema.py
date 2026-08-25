"""Pure-python self-check for the schema renderer — no bot deps needed."""
import ast
from pathlib import Path

# Parse bot.py instead of importing it (imports need telegram/anthropic/etc.)
tree = ast.parse(Path(__file__).with_name("bot.py").read_text())
ns = {}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == "format_schema":
        exec(compile(ast.Module([node], []), "bot.py", "exec"), ns)
format_schema = ns["format_schema"]


def col(table, name, data_type, udt_name=None, udt_schema="tractor"):
    return {"table_name": table, "column_name": name, "data_type": data_type,
            "udt_schema": udt_schema, "udt_name": udt_name or data_type}


COLUMNS = [
    col("animals", "id", "text"),
    col("animals", "sex", "USER-DEFINED", "AnimalSex"),
    col("animals", "tags", "ARRAY", "_text"),
    col("animals", "farm_id", "text"),
    col("animals", "secret_id", "text"),
    col("farms", "id", "text"),
    col("farms", "owner_sex", "USER-DEFINED", "AnimalSex"),
]
FKS = [
    {"tbl": "animals", "col": "farm_id", "ref_tbl": "farms", "ref_col": "id"},
    # points at a table the role can't see — must not leak into the prompt
    {"tbl": "animals", "col": "secret_id", "ref_tbl": "user_tokens", "ref_col": "id"},
]
ENUMS = [
    {"schema": "tractor", "name": "AnimalSex", "label": "MALE"},
    {"schema": "tractor", "name": "AnimalSex", "label": "FEMALE"},
    # same type name in another schema must not merge its labels in
    {"schema": "public", "name": "AnimalSex", "label": "OTHER"},
]

out = format_schema(COLUMNS, FKS, ENUMS)

assert set(out) == {"animals", "farms"}, out
animals = out["animals"]
assert animals.startswith("Table `animals`: "), animals
assert "sex (AnimalSex: MALE|FEMALE)" in animals, animals   # enum values inlined, in order
assert "OTHER" not in animals, animals                      # other schema's labels excluded
assert "farm_id (text -> farms.id)" in animals, animals     # FK target inlined
assert "secret_id (text)" in animals, animals               # FK to invisible table dropped
assert "user_tokens" not in animals, animals
assert "tags (text[])" in animals, animals                  # array rendered readably
assert "id (text)," in animals, animals                     # plain column unchanged

# values listed once: the second column using AnimalSex gets the bare type name
assert "owner_sex (AnimalSex)" in out["farms"], out["farms"]
assert "MALE" not in out["farms"], out["farms"]

# no enum metadata at all: fall back to the bare type name, don't crash
assert "sex (AnimalSex)" in format_schema(COLUMNS, [], [])["animals"]

print("ok")
