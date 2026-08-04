from bot import md_to_html

def test_bold():
    assert md_to_html("**Edad:** 19") == "<b>Edad:</b> 19"

def test_escape_html():
    assert md_to_html("peso < 500 & activo") == "peso &lt; 500 &amp; activo"

def test_inline_code():
    assert md_to_html("`physical_id`") == "<code>physical_id</code>"

def test_code_fence():
    assert md_to_html("```sql\nSELECT 1;\n```") == "<pre>SELECT 1;\n</pre>"

def test_header_to_bold():
    assert md_to_html("### Resumen") == "<b>Resumen</b>"

def test_unmatched_markers_stay_literal():
    assert md_to_html("5 * 3 y 2 ** algo") == "5 * 3 y 2 ** algo"

if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("md_to_html OK")
