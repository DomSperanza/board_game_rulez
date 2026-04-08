"""
Flask web UI: upload rule PDFs + chat over ingested rules. Streamlit remains in app.py.
"""
from __future__ import annotations

import hmac
import os
import sys
import warnings
from pathlib import Path

_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "None")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)

load_dotenv(_SRC.parent / ".env")
load_dotenv()

from chat_markdown import chat_html_from_markdown
from generation.gemini_client import get_answer
from ingestion.pipeline import delete_game_completely, ingest_uploaded_pdf
from ingestion.registry import list_library_games, sync_from_chroma_if_registry_empty
from retrieval.search import search_rulebook

app = Flask(
    __name__,
    template_folder=str(_SRC / "templates"),
    static_folder=str(_SRC / "static"),
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or "dev-change-me-in-production"


@app.template_filter("chat_md")
def _chat_md_filter(s: str | None):
    return chat_html_from_markdown(s or "")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


_THUMB_DIR = _ROOT / "data" / "game_thumbnails"


def _upload_password_required() -> bool:
    return bool(os.environ.get("INDEX_PASSWORD"))


def _index_password_ok() -> bool:
    expected = os.environ.get("INDEX_PASSWORD")
    if expected is None or expected == "":
        return True
    got = request.form.get("index_password") or ""
    return hmac.compare_digest(got.encode(), expected.encode())


def _games() -> list[dict]:
    sync_from_chroma_if_registry_empty()
    return list_library_games()


def _selected_display_name(games: list[dict]) -> str | None:
    names = [g["display_name"] for g in games]
    if not names:
        return None
    stored = session.get("playing_game")
    if stored in names:
        return stored
    return names[0]


@app.route("/", methods=["GET"])
def index():
    if "messages" not in session:
        session["messages"] = []
    games = _games()
    return render_template(
        "index.html",
        games=games,
        messages=session["messages"],
        selected_game=_selected_display_name(games),
        upload_requires_password=_upload_password_required(),
    )


@app.get("/game-thumb/<filename>")
def game_thumb(filename: str):
    if not filename.endswith(".jpg") or len(filename) != 20:
        abort(404)
    base = filename[:-4]
    if len(base) != 16 or any(c not in "0123456789abcdef" for c in base):
        abort(404)
    if not _THUMB_DIR.is_dir():
        abort(404)
    return send_from_directory(_THUMB_DIR, filename, max_age=86400)


@app.post("/upload")
def upload():
    if not _index_password_ok():
        flash("Wrong index password.", "error")
        return redirect(url_for("index"))
    game = request.form.get("game_name", "")
    file = request.files.get("pdf")
    try:
        complexity = int(request.form.get("complexity") or 5)
    except ValueError:
        complexity = 5
    complexity = max(1, min(10, complexity))
    ocr_sparse = request.form.get("ocr_sparse") == "1"
    ok, msg = ingest_uploaded_pdf(
        file, game, complexity=complexity, ocr_sparse_pages=ocr_sparse
    )
    flash(msg, "success" if ok else "error")
    if ok and game.strip():
        names = [g["display_name"] for g in _games()]
        display = " ".join(game.strip().split())
        if display in names:
            session["playing_game"] = display
            session.modified = True
    return redirect(url_for("index"))


@app.post("/remove-game")
def remove_game():
    if not _index_password_ok():
        flash("Wrong index password.", "error")
        return redirect(url_for("index"))
    game = request.form.get("game_name", "")
    ok, msg = delete_game_completely(game)
    flash(msg, "success" if ok else "error")
    if ok:
        display = " ".join(game.strip().split())
        if session.get("playing_game") == display:
            session.pop("playing_game", None)
            session.modified = True
    return redirect(url_for("index"))


@app.post("/ask")
def ask():
    if "messages" not in session:
        session["messages"] = []
    game = request.form.get("game_name", "")
    prompt = (request.form.get("prompt") or "").strip()
    games = _games()
    names = [g["display_name"] for g in games]
    if game not in names:
        flash("Pick a game from your library first.", "error")
        return redirect(url_for("index"))
    session["playing_game"] = game
    if not prompt:
        flash("Type a question to ask the rulebot.", "error")
        session.modified = True
        return redirect(url_for("index"))
    session["messages"].append({"role": "user", "content": prompt})
    chunks = search_rulebook(prompt, game)
    answer = get_answer(prompt, chunks)
    session["messages"].append({"role": "assistant", "content": answer})
    session.modified = True
    return redirect(url_for("index"))


@app.post("/clear")
def clear_chat():
    session["messages"] = []
    session.modified = True
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)
