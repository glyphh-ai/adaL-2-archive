"""THE PALETTE LAW — enforced.

Every color the TUI ever prints — the CLI banner, and every line the REPL
prints via Rich markup — must be: white #ffffff · grays #dadada/#a8a8a8/
#6c6c6c · lines #444444 · inks #121212/#1c1c1c/#262626 · and exactly ONE
purple #af5fff (plus any neutral gray). Nothing else — no blue, no green,
no red, no second purple, ever. Values must also be 256-safe: Terminal.app
has no truecolor, and tinted colors drift blue when quantized to the
256 cube.

Two layers of enforcement:
1. Source audit — every hex literal in ada/tui/*.py is in the law and
   survives 256-color quantization without changing hue.
2. Render audit — the REPL's actual dispatcher is driven against a fake
   backend and every color Rich renders to the console is checked
   against the law.
"""

import re
from pathlib import Path

from rich.color import Color, ColorSystem
from rich.console import Console

TUI = Path(__file__).resolve().parent.parent / "ada" / "tui"

PALETTE = {
    "#ffffff", "#dadada", "#a8a8a8", "#6c6c6c", "#444444", "#af5fff",
    "#121212", "#1c1c1c", "#262626", "#000000",
}


def _allowed(h: str) -> bool:
    h = h.lower()
    if h in PALETTE:
        return True
    r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
    if r == g == b:                      # neutral gray — cannot shift hue
        return True
    return False


def test_source_hexes_are_in_law_and_256_safe():
    for f in TUI.glob("*.py"):
        for h in set(re.findall(r"#[0-9a-fA-F]{6}", f.read_text())):
            assert _allowed(h), f"{f.name}: {h} is outside the palette law"
            r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
            d = Color.parse(h.lower()).downgrade(ColorSystem.EIGHT_BIT).get_truecolor()
            if r == g == b:
                assert d.red == d.green == d.blue, (
                    f"{f.name}: gray {h} quantizes off-neutral to {d.hex}")
            elif h.lower() in PALETTE:
                assert d.hex.lower() == h.lower(), (
                    f"{f.name}: accent {h} is not 256-exact (quantizes to {d.hex})")


def _painted_hexes(html: str) -> set[str]:
    return {h.lower() for h in re.findall(r"color:\s*(#[0-9a-fA-F]{6})", html)}


def test_repl_output_paints_only_the_law(monkeypatch):
    """Drive every command verb through the REPL dispatcher against a fake
    backend and assert every color Rich actually renders is in the law."""
    from ada.tui import repl as repl_mod

    record = Console(record=True, width=100, force_terminal=True,
                     color_system="truecolor", highlight=False)
    monkeypatch.setattr(repl_mod, "console", record)

    class FakeBackend:
        def call(self, tool: str, args: dict) -> dict:
            if tool == "stats":
                return {"count": 3, "versioned_keys": 1}
            if tool == "tell":
                return {"told": True, "version": 1}
            if tool == "ask":
                return {"answer": "brandi", "confidence": 0.9}
            if tool == "browse":
                return {"facts": [{"thought_id": "t1", "key": "wife.name",
                                   "version": 1, "content": "brandi"}]}
            if tool == "token_list":
                return {"tokens": [{"prefix": "ada_abc", "name": "cli",
                                    "permissions": ["read"], "status": "active",
                                    "expires_at": None}]}
            if tool == "merge_candidates":
                return {"candidates": [{"a": "x", "b": "y", "score": 0.8,
                                        "shared": ["k"], "conflicts": ["c"]}]}
            return {"error": "boom"}

    sess = repl_mod.Session("http://x/mcp")
    sess.backend = FakeBackend()

    for cmd in ("help", "stats", "tell hello world", "ask who am I",
                "list", "token", "merges", "fact somekey"):  # "fact" hits
        # FakeBackend's default {"error": "boom"} branch — exercises the
        # error-color path.
        repl_mod._dispatch(sess, cmd)

    hexes = _painted_hexes(record.export_html(inline_styles=True))
    bad = sorted(h for h in hexes if not _allowed(h))
    assert not bad, f"off-law painted colors: {bad}"
