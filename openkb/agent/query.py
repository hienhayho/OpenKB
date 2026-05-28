"""Q&A agent for querying the OpenKB knowledge base."""

from __future__ import annotations

import json
import os
from pathlib import Path

from agno.agent import Agent
from agno.models.message import Message
from agno.models.openai.like import OpenAILike
from agno.tools import tool

from openkb.agent.tools import (
    get_wiki_page_content,
    read_wiki_file,
    read_wiki_image,
    write_kb_file,
)

MAX_TURNS = 50
from openkb.schema import get_agents_md

_QUERY_INSTRUCTIONS_TEMPLATE = """\
You are OpenKB, a knowledge-base Q&A agent. Answer questions by retrieving wiki content \
using the tools below. Never answer from memory — always read the relevant pages first.

{schema_md}

## Tools
- `read_file(path)` — read any Markdown file relative to wiki root
- `get_page_content(doc_name, pages)` — fetch specific pages from a long (pageindex) document; \
  use tight ranges (e.g. "3-5,9"), never the whole doc
- `get_image(image_path)` — view an image referenced in source content

## When to call which tool

| Situation | Call |
|-----------|------|
| Start of every query | `read_file("index.md")` — always first |
| Need doc overview | `read_file("summaries/<doc>.md")` |
| Need cross-doc synthesis | `read_file("concepts/<slug>.md")` |
| Need full text of a short doc | `read_file(<full_text path from summary frontmatter>)` |
| Need specific pages of a long doc | `get_page_content(<doc_name>, <pages>)` — check summary tree for page ranges |
| Answer mentions a figure/chart | `get_image(<path from source content>)` |

## Rules
- ALWAYS call `read_file("index.md")` first — it maps all docs and concepts.
- Read summaries before source content; source only when summary is insufficient.
- For pageindex docs use `get_page_content`, never `read_file` on the source JSON.
- Stop retrieving once you have enough to answer — do not fetch speculatively.
- Cite sources as [[summaries/doc]] or [[concepts/slug]] inline in your answer.
- If information is not in the wiki, say so — do not invent.
"""


def _make_model(kb_dir: Path | None = None) -> OpenAILike:
    """Build OpenAILike from KB config or env vars."""
    import os
    from openkb.config import load_config

    cfg: dict = {}
    if kb_dir is not None:
        config_path = kb_dir / ".openkb" / "config.yaml"
        if config_path.exists():
            cfg = load_config(config_path)

    model_id = cfg.get("openai_model_name") or os.environ.get(
        "OPENAI_MODEL_NAME", "gpt-4o-mini"
    )
    base_url = cfg.get("openai_base_url") or os.environ.get(
        "OPENAI_BASE_URL", "https://api.openai.com/v1"
    )
    api_key = cfg.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")
    _extra_body_env = os.environ.get("OPENAI_EXTRA_BODY", "")
    extra_body = cfg.get("openai_extra_body") or (
        json.loads(_extra_body_env) if _extra_body_env else {}
    )

    return OpenAILike(
        id=model_id,
        base_url=base_url,
        api_key=api_key,
        extra_body=extra_body,
    )


def build_query_agent(
    wiki_root: str, model: str, language: str = "en", kb_dir: Path | None = None
) -> Agent:
    """Build and return the Q&A agent."""
    schema_md = get_agents_md(Path(wiki_root))
    instructions = _QUERY_INSTRUCTIONS_TEMPLATE.format(schema_md=schema_md)
    instructions += f"\n\nIMPORTANT: Answer in {language} language."

    @tool
    def read_file(path: str) -> str:
        """Read a Markdown file from the wiki.
        Args:
            path: File path relative to wiki root (e.g. 'summaries/paper.md').
        """
        return read_wiki_file(path, wiki_root)

    @tool
    def get_page_content(doc_name: str, pages: str) -> str:
        """Get text content of specific pages from a PageIndex (long) document.
        Only use for documents with doc_type: pageindex. For short documents,
        use read_file instead.
        Args:
            doc_name: Document name (e.g. 'attention-is-all-you-need').
            pages: Page specification (e.g. '3-5,7,10-12').
        """
        return get_wiki_page_content(doc_name, pages, wiki_root)

    @tool
    def get_image(image_path: str) -> str:
        """View an image from the wiki.

        Use when a question asks about a specific figure, chart, or diagram
        you'd need to see to answer accurately.

        Args:
            image_path: Image path relative to wiki root (e.g. 'sources/images/doc/p1_img1.png').
        """
        result = read_wiki_image(image_path, wiki_root)
        if result["type"] == "image":
            return result["image_url"]
        return result["text"]

    return Agent(
        name="wiki-query",
        instructions=instructions,
        tools=[read_file, get_page_content, get_image],
        model=_make_model(kb_dir),
        debug_mode=True,
    )


def build_chat_agent(
    kb_dir: Path,
    model: str,
    language: str = "en",
) -> Agent:
    """Build the chat agent: query agent + a write tool restricted to
    ``<kb>/wiki/explorations/**`` and ``<kb>/output/**``.
    """
    wiki_root = str(kb_dir / "wiki")
    kb_root = str(kb_dir)
    base = build_query_agent(wiki_root, model, language=language, kb_dir=kb_dir)

    @tool
    def write_file(path: str, content: str) -> str:
        """Write a text file under the KB.

        Allowed paths (relative to KB root):
          * ``wiki/explorations/**`` — chat-derived notes.
          * ``output/**``            — generator artifacts (skills, etc.).

        Any other path is rejected. Parent directories are created.

        Args:
            path: File path relative to KB root
                (e.g. ``"output/skills/demo/SKILL.md"``).
            content: Full text content to write (overwrites if file exists).
        """
        return write_kb_file(path, content, kb_root)

    return base.clone(tools=[*base.tools, write_file])


async def run_query(
    question: str,
    kb_dir: Path,
    model: str,
    stream: bool = False,
    *,
    raw: bool = False,
) -> str:
    """Run a Q&A query against the knowledge base."""
    import sys
    from openkb.config import load_config

    openkb_dir = kb_dir / ".openkb"
    config = load_config(openkb_dir / "config.yaml")
    language: str = config.get("language", "en")

    wiki_root = str(kb_dir / "wiki")
    agent = build_query_agent(wiki_root, model, language=language, kb_dir=kb_dir)

    if not stream:
        result = await agent.arun(input=Message(role="user", content=question))
        return result.content or ""

    import os

    use_color = sys.stdout.isatty() and not os.environ.get("NO_COLOR", "")

    from openkb.agent.chat import (
        _build_style,
        _fmt,
        _format_tool_line,
        _make_markdown,
        _make_rich_console,
    )

    style = _build_style(use_color)

    from rich.live import Live

    if use_color and not raw:
        console = _make_rich_console()
    else:
        console = None  # type: ignore[assignment]

    def _start_live() -> Live | None:
        if console is None:
            return None
        lv = Live(console=console, vertical_overflow="visible")
        lv.start()
        return lv

    live: Live | None = None
    last_was_text = False
    need_blank_before_text = False
    collected: list[str] = []
    segment: list[str] = []

    try:
        live = _start_live()
        async for event in agent.arun(
            input=Message(role="user", content=question),
            stream=True,
            stream_events=True,
        ):
            # agno RunResponseEvent: check content vs tool events
            content = getattr(event, "content", None)
            if content and isinstance(content, str):
                if need_blank_before_text:
                    if console is not None:
                        print()
                        segment = []
                        live = _start_live()
                    else:
                        sys.stdout.write("\n")
                    need_blank_before_text = False
                collected.append(content)
                segment.append(content)
                last_was_text = True
                if live and "\n" in content:
                    joined = "".join(segment)
                    visible = joined[: joined.rfind("\n") + 1]
                    if visible:
                        live.update(_make_markdown(visible))
                elif not live:
                    sys.stdout.write(content)
                    sys.stdout.flush()
            else:
                # Tool call event
                tool_name = getattr(event, "tool_name", None) or getattr(
                    event, "event", ""
                )
                if tool_name and last_was_text:
                    if live:
                        if segment:
                            live.update(_make_markdown("".join(segment)))
                        live.stop()
                        live = None
                    else:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                    last_was_text = False
                    tool_args = str(getattr(event, "tool_args", "") or "")
                    _fmt(
                        style,
                        ("class:tool", _format_tool_line(tool_name, tool_args) + "\n"),
                    )
                    need_blank_before_text = True
    finally:
        if live:
            if segment:
                live.update(_make_markdown("".join(segment)))
            live.stop()
        print()

    return "".join(collected) if collected else ""
