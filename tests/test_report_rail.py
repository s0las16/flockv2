"""QA tests for the report geo rail — sundial.report.render_html(adj, out, extra).

`extra` is the optional geo-rail payload bag handed to render_html. Per the
spec under test (see matrix below) the recognised keys are:

    map_svg      — inline-SVG map, PRIMARY map payload (trusted, verbatim)
    map_fragment — trusted HTML map fragment (verbatim; below map_svg)
    map_static   — base64 PNG, last-resort -> <img src="data:image/png;base64,…">
    loc_summary  — {"best_name": str, "ranked": [{cc, score}, …]} — UNTRUSTED,
                   every field must be HTML-escaped before insertion
    chain_html   — trusted reasoning-chain fragment (verbatim)

TRUST-BOUNDARY NOTE (documented design, asserted by
test_trusted_keys_insert_verbatim_by_design): map_svg / map_fragment /
chain_html are inserted VERBATIM — the producers (sundial.map_viz,
sundial.osint_chain, sundial.vec_map) are trusted and must sanitize.
loc_summary on the other hand is untrusted data and MUST be escaped.

Adjudications are fabricated through the real pipeline helpers:
sundial.cli._demo_pack() + sundial.adjudicate.adjudicate (deterministic
rule-based fallback — LLM env vars are scrubbed inside _adjudicate()).

stdlib only. Run either:
    python3 -m pytest tests/test_report_rail.py
    python3 tests/test_report_rail.py        # built-in fallback runner
"""
import os
import re
import sys
import tempfile
import traceback
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sundial.cli import _demo_pack            # noqa: E402
from sundial.adjudicate import adjudicate     # noqa: E402
from sundial.report import render_html        # noqa: E402
from sundial.schema import ClaimVerdict, Verdict  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _adjudicate(pack):
    """adjudicate() forced onto the deterministic rule-based fallback so the
    verdict does not depend on a reachable W&B/OpenAI endpoint. Env restored."""
    saved = {}
    for k in ("WANDB_API_KEY", "WANDB_BASE_URL", "WANDB_MODEL"):
        if k in os.environ:
            saved[k] = os.environ.pop(k)
    try:
        return adjudicate(pack)
    finally:
        os.environ.update(saved)


def _demo_adj():
    return _adjudicate(_demo_pack())


def _supports_pack():
    """Demo pack with every signal flipped to SUPPORTS -> SUPPORTED verdict."""
    pack = _demo_pack()
    for ev in pack.evidence:
        ev.verdict = ClaimVerdict.SUPPORTS
    return pack


def _doc(adj, extra=None):
    """Render to a temp file; return the document text."""
    with tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "report.html")
        assert render_html(adj, out, extra) == out
        return Path(out).read_text(encoding="utf-8")


def _map_region(doc):
    """First ~300 chars of rail content after the rail title — for diagnostics."""
    m = re.search(r'<div class="railtitle">.*?</div>(.{0,300})', doc, re.S)
    return m.group(1) if m else "<no railtitle block found>"


class _Tree(HTMLParser):
    """Balance + parentage audit of the rendered document.

    HTMLParser already treats <script>/<style> bodies as CDATA, so only
    real markup is checked. Void elements are never pushed; self-closing
    (<x/>) tags push+pop via handle_startendtag.
    """

    _VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input",
             "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []               # [(tag, class_attr)] currently open
        self.errors = []              # nesting faults
        self.rail_in_layout = False   # aside.rail is a child of div.layout
        self.panel_in_rail = False    # div.railpanel inside aside.rail
        self.card_in_layout = False   # div.card inside div.layout
        self.layout_in_wrap = False   # div.layout inside div.wrap

    def _has(self, tag, cls):
        return any(t == tag and cls in c.split() for t, c in self.stack)

    def handle_starttag(self, tag, attrs):
        if tag in self._VOID:
            return
        cls = dict(attrs).get("class") or ""
        classes = set(cls.split())
        # parentage is checked BEFORE pushing self onto the stack
        if tag == "aside" and "rail" in classes:
            self.rail_in_layout = self._has("div", "layout")
        if tag == "div" and "railpanel" in classes:
            self.panel_in_rail = self._has("aside", "rail")
        if tag == "div" and "card" in classes:
            self.card_in_layout = self._has("div", "layout")
        if tag == "div" and "layout" in classes:
            self.layout_in_wrap = self._has("div", "wrap")
        self.stack.append((tag, cls))

    def handle_endtag(self, tag):
        if tag in self._VOID:
            return
        if self.stack and self.stack[-1][0] == tag:
            self.stack.pop()
            return
        names = [t for t, _ in self.stack]
        if tag in names:  # recoverable: some open element was never closed
            idx = len(names) - 1 - names[::-1].index(tag)
            self.errors.append(
                f"</{tag}> closed over unclosed elements {self.stack[idx + 1:]}")
            del self.stack[idx:]
        else:
            self.errors.append(f"stray </{tag}> with no open element")


# --------------------------------------------------------------------------- #
# matrix 1 — extra=None: graceful "no geo read" empty state, rail still there
# --------------------------------------------------------------------------- #
def test_extra_none_renders_nogeo_empty_state():
    doc = _doc(_demo_adj())  # extra=None
    assert '<aside class="rail">' in doc, "rail aside missing entirely"
    assert 'class="nogeo"' in doc, \
        f"expected nogeo empty-state; rail shows: {_map_region(doc)}"
    assert "no geo read" in doc
    assert 'class="mapbox"' not in doc, "map block rendered despite no geo data"
    # no <svg> map is required for a complete page — document must be whole
    assert doc.startswith("<!DOCTYPE html>")
    assert doc.rstrip().endswith("</html>")


# --------------------------------------------------------------------------- #
# matrix 2 — map_svg inline + loc_summary readout + ranked bars
# --------------------------------------------------------------------------- #
def test_map_svg_renders_inline_in_mapbox():
    extra = {"map_svg": "<svg viewBox='0 0 10 10'><rect/></svg>",
             "loc_summary": {"best_name": "Germany",
                             "ranked": [{"cc": "DE", "score": 0.7}]}}
    doc = _doc(_demo_adj(), extra)
    assert 'class="mapbox"' in doc, \
        f"expected .mapbox wrapper; rail shows: {_map_region(doc)}"
    assert "<svg viewBox='0 0 10 10'><rect/></svg>" in doc, \
        f"expected map_svg inserted verbatim (trusted); rail shows: {_map_region(doc)}"


def test_loc_summary_renders_best_name_and_ranked_bars():
    extra = {"map_svg": "<svg viewBox='0 0 10 10'><rect/></svg>",
             "loc_summary": {"best_name": "Germany",
                             "ranked": [{"cc": "DE", "score": 0.7}]}}
    doc = _doc(_demo_adj(), extra)
    assert '<div class="locbest">Germany</div>' in doc, "best_name readout missing"
    assert '<span class="loccc">DE</span>' in doc, "ranked cc label missing"
    assert re.search(r'class="locfill" style="width:70%"', doc), \
        "expected confidence bar at 70% for score 0.7"
    assert '<span class="locscore">0.70</span>' in doc, "numeric score missing"


# --------------------------------------------------------------------------- #
# matrix 3 — priority: map_svg beats map_fragment when both are supplied
# --------------------------------------------------------------------------- #
def test_map_svg_wins_over_map_fragment():
    extra = {"map_svg": "<svg viewBox='0 0 10 10'><rect/></svg>",
             "map_fragment": "<div>frag</div>"}
    doc = _doc(_demo_adj(), extra)
    assert "<svg viewBox='0 0 10 10'><rect/></svg>" in doc, \
        f"expected map_svg to win priority; rail shows: {_map_region(doc)}"
    assert "<div>frag</div>" not in doc, \
        "map_fragment rendered although map_svg should take priority"


def test_map_fragment_beats_map_static():
    """Mid-priority rung: fragment preferred over the static PNG fallback."""
    extra = {"map_fragment": "<div>frag</div>", "map_static": "aGVsbG8="}
    doc = _doc(_demo_adj(), extra)
    assert "<div>frag</div>" in doc
    assert '<img class="mapimg"' not in doc, \
        "map_static rendered although map_fragment should take priority"


# --------------------------------------------------------------------------- #
# matrix 4 — map_static: base64 PNG becomes an <img> data URI
# --------------------------------------------------------------------------- #
def test_map_static_renders_base64_img():
    doc = _doc(_demo_adj(), {"map_static": "aGVsbG8="})
    m = re.search(r'<img[^>]+src="data:image/png;base64,aGVsbG8="', doc)
    assert m, f"expected <img …data:image/png;base64,aGVsbG8=>; rail shows: {_map_region(doc)}"
    assert 'class="nogeo"' not in doc, "empty state shown despite map_static"
    assert 'class="mapbox"' in doc


# --------------------------------------------------------------------------- #
# matrix 5 — chain_html produces the reasoning-chain block, verbatim
# --------------------------------------------------------------------------- #
def test_chain_html_renders_chainbox():
    doc = _doc(_demo_adj(), {"chain_html": "<ol><li>step</li></ol>"})
    assert 'class="chainbox"' in doc, "chainbox container missing"
    assert "<ol><li>step</li></ol>" in doc, "chain fragment not verbatim"
    assert "reasoning chain" in doc, "chainbox subheading missing"


# --------------------------------------------------------------------------- #
# matrix 6 — hostile input: loc_summary fields MUST be escaped; trusted keys
# (map_svg / map_fragment / chain_html) are verbatim BY DESIGN (documented)
# --------------------------------------------------------------------------- #
def test_loc_summary_fields_are_escaped():
    evil = {"best_name": "<script>alert(1)</script>",
            "ranked": [{"cc": "<img src=x onerror=alert(2)>", "score": 0.9}]}
    doc = _doc(_demo_adj(), {"loc_summary": evil})
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in doc, \
        "best_name not escaped — XSS via loc_summary"
    assert "&lt;img src=x onerror=alert(2)&gt;" in doc, \
        "ranked cc not escaped — XSS via loc_summary.ranked"
    assert "<script>alert(1)</script>" not in doc
    assert "<img src=x onerror=alert(2)>" not in doc
    # invariant: the only raw <script> is the report's own animation block
    assert doc.count("<script") == 1, \
        f"expected 1 raw <script> (report JS), found {doc.count('<script')}"
    assert 'class="locbest"' in doc


def test_map_static_src_is_attribute_escaped():
    """map_static lands inside src="…" — an attribute-injection attempt must
    stay entity-escaped so it cannot break out of the attribute."""
    doc = _doc(_demo_adj(), {'map_static': 'x"><script>alert(9)</script>'})
    assert "<script>alert(9)</script>" not in doc
    assert doc.count("<script") == 1
    assert "&quot;&gt;&lt;script&gt;" in doc, "map_static src not escaped"


def test_empty_valued_loc_summary_shows_map_placeholder():
    """BEHAVIOUR PIN (possible minor defect): loc_summary={best_name:'',
    ranked:[]} is a non-empty dict so has_geo=True — the rail shows the bare
    'map unavailable' placeholder with no readout, NOT the 'no geo read'
    empty state that an equally information-free loc_summary={} produces.
    Reported as a low-severity UX inconsistency; flip the assertion if the
    renderer is fixed to inspect loc values, not just dict truthiness."""
    doc = _doc(_demo_adj(), {"loc_summary": {"best_name": "", "ranked": []}})
    assert '<div class="mapnone">map unavailable</div>' in doc
    assert 'class="nogeo"' not in doc  # current behaviour — see docstring


def test_trusted_keys_insert_verbatim_by_design():
    """DESIGN DOCUMENTATION — map_fragment / chain_html (and, per spec,
    map_svg) are TRUSTED pipelines inserted UNESCAPED. This test pins the
    contract: markup survives verbatim, producers sanitize upstream."""
    frag = "<div class='frag'><b>bold</b></div>"
    chain = "<ol><li><em>emph</em></li></ol>"
    doc = _doc(_demo_adj(), {"map_fragment": frag, "chain_html": chain})
    assert frag in doc, "map_fragment was escaped — violates verbatim contract"
    assert chain in doc, "chain_html was escaped — violates verbatim contract"
    assert "&lt;b&gt;bold&lt;/b&gt;" not in doc


# --------------------------------------------------------------------------- #
# matrix 7 — garbage extra values: no crash, graceful degradation
# --------------------------------------------------------------------------- #
def test_garbage_extra_values_do_not_crash():
    doc = _doc(_demo_adj(),
               {"map_svg": 123, "loc_summary": "junk", "ranked": None})
    assert doc.startswith("<!DOCTYPE html>")
    assert doc.rstrip().endswith("</html>")
    assert '<aside class="rail">' in doc
    # nothing usable was supplied -> the empty state is the graceful fallback
    assert 'class="nogeo"' in doc, \
        f"expected nogeo fallback for garbage extra; rail shows: {_map_region(doc)}"


def test_non_dict_extra_and_junk_fields_do_not_crash():
    for junk_extra in ("junk", 42, ["x"], {"loc_summary": {"best_name": 123,
                                                           "ranked": "junk"}}):
        doc = _doc(_demo_adj(), junk_extra)
        assert doc.startswith("<!DOCTYPE html>"), junk_extra
        assert doc.rstrip().endswith("</html>"), junk_extra
        assert '<aside class="rail">' in doc, junk_extra


# --------------------------------------------------------------------------- #
# matrix 8 — full-document invariants: one <script>, valid .rail/.layout
# nesting, no unclosed <div> in the rail
# --------------------------------------------------------------------------- #
def test_single_script_tag_invariant():
    extras = [None,
              {"map_static": "aGVsbG8="},
              {"loc_summary": {"best_name": "X",
                               "ranked": [{"cc": "DE", "score": 0.5}]}},
              {"chain_html": "<ol><li>s</li></ol>"},
              {"map_fragment": "<div>frag</div>"}]
    for extra in extras:
        doc = _doc(_demo_adj(), extra)
        assert doc.count("<script") == 1, \
            f"extra={extra!r}: expected exactly 1 <script>, got {doc.count('<script')}"
        assert doc.count("</script>") == 1, \
            f"extra={extra!r}: expected exactly 1 </script>, got {doc.count('</script>')}"


def test_layout_rail_nesting_is_balanced():
    extras = [None,
              {"map_static": "aGVsbG8="},
              {"loc_summary": {"best_name": "Germany",
                               "ranked": [{"cc": "DE", "score": 0.7}]}},
              {"chain_html": "<ol><li>s</li></ol>"},
              {"map_fragment": "<div>frag</div>"},
              {"map_svg": "<svg viewBox='0 0 10 10'><rect/></svg>"}]
    for extra in extras:
        doc = _doc(_demo_adj(), extra)
        tree = _Tree()
        tree.feed(doc)
        tree.close()
        assert not tree.errors, f"extra={extra!r}: nesting faults {tree.errors}"
        assert not tree.stack, \
            f"extra={extra!r}: unclosed elements at EOF: {tree.stack}"
        assert tree.layout_in_wrap, "div.layout not inside div.wrap"
        assert tree.card_in_layout, "div.card not inside div.layout"
        assert tree.rail_in_layout, "aside.rail not inside div.layout"
        assert tree.panel_in_rail, "div.railpanel not inside aside.rail"
        # no unclosed <div> inside the rail specifically
        rail = doc.split('<aside class="rail">', 1)[1].split('</aside>', 1)[0]
        assert rail.count("<div") == rail.count("</div>"), \
            f"extra={extra!r}: rail divs unbalanced " \
            f"({rail.count('<div')} open vs {rail.count('</div>')} close)"


# --------------------------------------------------------------------------- #
# matrix 9 — verdict variants drive accent colour + badge text
# --------------------------------------------------------------------------- #
def test_verdict_variants_change_accent_and_badge():
    contra = _adjudicate(_demo_pack())
    supp = _adjudicate(_supports_pack())
    assert contra.verdict is Verdict.CONTRADICTED, \
        f"demo pack should adjudicate CONTRADICTED, got {contra.verdict}"
    assert supp.verdict is Verdict.SUPPORTED, \
        f"all-supports pack should adjudicate SUPPORTED, got {supp.verdict}"

    d_c, d_s = _doc(contra), _doc(supp)
    assert "--accent:#f87171" in d_c, "CONTRADICTED accent (#f87171) missing"
    assert "--accent:#4ade80" in d_s, "SUPPORTED accent (#4ade80) missing"
    assert d_c.split("--accent:")[1][:7] != d_s.split("--accent:")[1][:7], \
        "accent colours identical across verdicts"
    assert 'id="verdictTxt" data-final="CONTRADICTED"' in d_c
    assert 'id="verdictTxt" data-final="SUPPORTED"' in d_s
    assert "<title>Sundial — CONTRADICTED</title>" in d_c
    assert "<title>Sundial — SUPPORTED</title>" in d_s


# --------------------------------------------------------------------------- #
# matrix 10 — mobile: @media query collapses .layout to a single column
# --------------------------------------------------------------------------- #
def test_mobile_media_query_collapses_layout():
    doc = _doc(_demo_adj())
    m = re.search(r"@media \(max-width:(\d+)px\)[^{]*\{(.*?)\n\}", doc, re.S)
    assert m, "no @media (max-width:…px) block found — no mobile breakpoint"
    block = m.group(2)
    assert re.search(r"\.layout\s*\{[^}]*grid-template-columns:\s*minmax\(0,1fr\)",
                     block), \
        f"media block does not collapse .layout to one column: {block!r}"
    assert re.search(r"\.rail\s*\{[^}]*position:\s*static", block), \
        "media block should un-stick the rail (position:static)"


# --------------------------------------------------------------------------- #
# direct-run fallback runner (pytest runs the test_* functions natively)
# --------------------------------------------------------------------------- #
def _run():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} ok")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _run()
