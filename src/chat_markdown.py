"""Turn model Markdown into safe HTML for Flask chat bubbles."""
from __future__ import annotations

import bleach
import markdown
from markupsafe import Markup

_TAGS = frozenset({
    "p",
    "br",
    "strong",
    "em",
    "b",
    "i",
    "ul",
    "ol",
    "li",
    "code",
    "pre",
    "blockquote",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",
})


def chat_html_from_markdown(text: str) -> Markup:
    if not text:
        return Markup("")
    raw = markdown.markdown(
        text,
        extensions=["extra", "nl2br", "sane_lists"],
        output_format="html5",
    )
    clean = bleach.clean(raw, tags=list(_TAGS), attributes={}, strip=True)
    return Markup(clean)
