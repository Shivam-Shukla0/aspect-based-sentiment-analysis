"""Checks for the static frontend in ``app/static`` (structure and client logic)."""

import json
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"


class _Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))


def _tags() -> list[tuple[str, dict[str, str | None]]]:
    collector = _Collector()
    collector.feed((STATIC / "index.html").read_text(encoding="utf-8"))
    return collector.tags


def _find(tag: str, **attrs: str) -> list[dict[str, str | None]]:
    return [
        found
        for name, found in _tags()
        if name == tag and all(found.get(key) == value for key, value in attrs.items())
    ]


def test_page_has_viewport_for_narrow_screens() -> None:
    assert _find("meta", name="viewport")


def test_page_has_review_input_and_form() -> None:
    assert _find("form", id="review-form")
    assert _find("textarea", id="review")


def test_page_offers_a_model_selector() -> None:
    values = {box["value"] for box in _find("input", name="model", type="checkbox")}
    assert values == {"lstm", "bert", "llm"}


def test_page_links_its_assets() -> None:
    assert _find("link", rel="stylesheet", href="style.css")
    assert _find("script", src="app.js")
    assert (STATIC / "style.css").is_file()
    assert (STATIC / "app.js").is_file()


def test_page_has_a_narrow_screen_rule() -> None:
    assert "@media (max-width" in (STATIC / "style.css").read_text(encoding="utf-8")


def test_script_never_inserts_markup() -> None:
    source = (STATIC / "app.js").read_text(encoding="utf-8")
    for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert forbidden not in source


NODE = shutil.which("node")

CLIENT_CHECK = """
const app = require(process.argv[2]);
const out = {};
out.empty = app.validateReview("   ");
out.long = app.validateReview("x".repeat(2001));
out.ok = app.validateReview("Great battery");
const answer = (status, body) => async () => ({
  ok: status < 400,
  status,
  json: async () => body,
});
(async () => {
  global.fetch = answer(200, { predictions: [{ aspect: "battery", sentiment: "positive" }] });
  out.predictions = await app.predict("Great battery", "lstm");
  const failures = {
    badRequest: answer(400, { error: "review must not be empty" }),
    noBody: async () => ({ ok: false, status: 503, json: async () => { throw new Error("x"); } }),
    wrongShape: answer(200, { nothing: true }),
    network: async () => { throw new TypeError("offline"); },
  };
  for (const [name, handler] of Object.entries(failures)) {
    global.fetch = handler;
    try {
      await app.predict("Great battery", "lstm");
      out[name] = "no error";
    } catch (error) {
      out[name] = error.message;
    }
  }
  console.log(JSON.stringify(out));
})();
"""


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_client_validation_and_error_handling(tmp_path: Path) -> None:
    script = tmp_path / "check.js"
    script.write_text(CLIENT_CHECK, encoding="utf-8")
    assert NODE is not None
    result = subprocess.run(  # noqa: S603  (resolved node binary, fixed script)
        [NODE, str(script), str(STATIC / "app.js")], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["empty"] == "Enter a review first."
    assert "2000" in out["long"]
    assert out["ok"] == ""
    assert out["predictions"] == [{"aspect": "battery", "sentiment": "positive"}]
    assert out["badRequest"] == "review must not be empty"
    assert out["noBody"] == "HTTP 503"
    assert out["wrongShape"] == "The server sent an unexpected response."
    assert out["network"] == "Could not reach the server."
