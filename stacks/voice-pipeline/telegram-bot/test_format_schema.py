"""Pure-python self-check for the schema renderer — no bot deps needed."""
import ast
from pathlib import Path

# Parse bot.py instead of importing it (imports need telegram/anthropic/etc.)
tree = ast.parse(Path(__file__).with_name("bot.py").read_text())
wanted = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == "format_schema":
        wanted.append(node)
    elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "TYPE_ABBREV" for t in node.targets):
        wanted.insert(0, node)  # the renderer reads it at call time
ns = {}
exec(compile(ast.Module(wanted, []), "bot.py", "exec"), ns)
format_schema = ns["format_schema"]


def col(table, name, data_type, udt_name=None, udt_schema="tractor"):
    return {"table_name": table, "column_name": name, "data_type": data_type,
            "udt_schema": udt_schema, "udt_name": udt_name or data_type}


COLUMNS = [
    col("animals", "id", "uuid"),
    col("animals", "created_at", "timestamp without time zone"),
    col("animals", "name", "character varying"),
    col("animals", "sex", "USER-DEFINED", "AnimalSex"),
    col("animals", "tags", "ARRAY", "_text"),
    col("animals", "farm_id", "uuid"),
    col("animals", "secret_id", "uuid"),
    col("farms", "id", "uuid"),
    col("farms", "created_at", "timestamp without time zone"),
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

preamble, out = format_schema(COLUMNS, FKS, ENUMS)

assert set(out) == {"animals", "farms"}, out
animals = out["animals"]
assert animals.startswith("Table `animals`: "), animals
assert "sex (AnimalSex: MALE|FEMALE)" in animals, animals   # enum values inlined, in order
assert "OTHER" not in animals, animals                      # other schema's labels excluded
assert "user_tokens" not in animals, animals

# --- A1: information_schema type names abbreviated
assert "name (str)" in animals, animals                     # character varying -> str
assert "character varying" not in animals, animals
assert "tags (text[])" in animals, animals                  # array still readable

# --- A2: the arrow implies the type, so the FK column carries no type
assert "farm_id (-> farms.id)" in animals, animals
assert "farm_id (uuid" not in animals, animals
# FK to an invisible table degrades to a normal typed column
assert "secret_id (uuid)" in animals, animals

# --- A3: columns every table shares are hoisted into the preamble, once
assert preamble.startswith("Every table also has, not repeated below: "), preamble
assert "id (uuid)" in preamble, preamble
assert "created_at (ts)" in preamble, preamble              # abbreviated there too
assert preamble.endswith("\n"), repr(preamble)              # joins straight onto the schema
def names(line):  # "Table `x`: a (int), b (-> y.z)" -> {"a", "b"}
    return {p.split(" (")[0] for p in line.split(": ", 1)[1].split(", ")}

for line in out.values():
    assert "id" not in names(line), line                    # ...and not repeated per table
    assert "created_at" not in names(line), line
# a column only some tables have must stay inline
assert "secret_id" in animals and "secret_id" not in preamble, (animals, preamble)

# values listed once: the second column using AnimalSex gets the bare type name
assert "owner_sex (AnimalSex)" in out["farms"], out["farms"]
assert "MALE" not in out["farms"], out["farms"]

# no enum metadata at all: fall back to the bare type name, don't crash
assert "sex (AnimalSex)" in format_schema(COLUMNS, [], [])[1]["animals"]

# --- hoisting guards
# one table only: nothing is "shared", so nothing is hoisted
solo_pre, solo = format_schema([col("farms", "id", "uuid")], [], [])
assert solo_pre == "", solo_pre
assert "id (uuid)" in solo["farms"], solo

# hoisting must never leave a table with an empty column list
thin_pre, thin = format_schema([col("animals", "id", "uuid"), col("animals", "name", "text"),
                                col("farms", "id", "uuid")], [], [])
assert thin_pre == "", thin_pre                             # farms would have been emptied
assert "id (uuid)" in thin["farms"], thin

print("ok")
