"""PageIndex indexer for long documents."""
from __future__ import annotations

import json as json_mod
import logging
import os

from dataclasses import dataclass
from pathlib import Path

from pageindex import PageIndexClient

from openkb.config import load_config
from openkb.tree_renderer import render_summary_md

logger = logging.getLogger(__name__)


@dataclass
class IndexResult:
    """Result of indexing a long document via PageIndex."""

    doc_id: str
    description: str
    tree: dict


def index_long_document(pdf_path: Path, kb_dir: Path) -> IndexResult:
    """Index a long PDF document using PageIndex and write wiki pages."""
    openkb_dir = kb_dir / ".openkb"
    config = load_config(openkb_dir / "config.yaml")

    model: str = config.get("openai_model_name") or os.environ.get("OPENAI_MODEL_NAME", config.get("model", "gpt-4o-mini"))
    api_key: str = config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")
    base_url: str = config.get("openai_base_url") or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    extra_body_env = os.environ.get("OPENAI_EXTRA_BODY", "")
    extra_body: dict = config.get("openai_extra_body") or (
        json_mod.loads(extra_body_env) if extra_body_env else {}
    )

    client = PageIndexClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        extra_body=extra_body or None,
        workspace=str(openkb_dir),
    )

    doc_id = client.index(str(pdf_path))
    logger.info("PageIndex indexed %s → doc_id=%s", pdf_path.name, doc_id)

    # get_document() returns lightweight meta only (no structure).
    # Load full doc from workspace file to get structure + summaries.
    workspace_file = openkb_dir / f"{doc_id}.json"
    if workspace_file.exists():
        full_doc = json_mod.loads(workspace_file.read_text(encoding="utf-8"))
    else:
        doc_json = client.get_document(doc_id)
        full_doc = json_mod.loads(doc_json) if isinstance(doc_json, str) else doc_json

    doc_name: str = full_doc.get("doc_name", pdf_path.stem)
    description: str = full_doc.get("doc_description", "")
    structure: list = full_doc.get("structure", [])

    tree = {
        "doc_name": doc_name,
        "doc_description": description,
        "structure": structure,
    }

    # Write wiki/sources/ — per-page content
    sources_dir = kb_dir / "wiki" / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    images_dir = sources_dir / "images" / pdf_path.stem

    all_pages: list = []

    # Try fetching pages from PageIndexClient (already parsed during index())
    try:
        from openkb.converter import get_pdf_page_count
        page_count = get_pdf_page_count(pdf_path)
        raw = client.get_page_content(doc_id, f"1-{page_count}")
        pages_data = json_mod.loads(raw) if isinstance(raw, str) else raw
        if isinstance(pages_data, list):
            all_pages = pages_data
    except Exception as exc:
        logger.warning("get_page_content failed for %s: %s; falling back to pymupdf", pdf_path.name, exc)

    if not all_pages:
        from openkb.images import convert_pdf_to_pages
        all_pages = convert_pdf_to_pages(pdf_path, pdf_path.stem, images_dir)

    (sources_dir / f"{pdf_path.stem}.json").write_text(
        json_mod.dumps(all_pages, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # Write wiki/summaries/
    summaries_dir = kb_dir / "wiki" / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    summary_md = render_summary_md(tree, pdf_path.stem, doc_id)
    (summaries_dir / f"{pdf_path.stem}.md").write_text(summary_md, encoding="utf-8")

    return IndexResult(doc_id=doc_id, description=description, tree=tree)
