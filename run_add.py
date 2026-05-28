"""
Run add_single_file programmatically against samples/Blog_RAG.pdf.

Usage:
    uv run python run_add.py
    OPENAI_API_KEY=sk-... OPENAI_MODEL_NAME=gpt-4o OPENAI_BASE_URL=https://api.openai.com/v1 uv run python run_add.py

Env vars (all optional, have defaults):
    OPENAI_MODEL_NAME   — model id           (default: gpt-4o-mini)
    OPENAI_BASE_URL     — API base URL       (default: https://api.openai.com/v1)
    OPENAI_API_KEY      — API key
    OPENAI_EXTRA_BODY   — JSON string        (default: {})
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from openkb.schema import AGENTS_MD
from openkb.config import DEFAULT_CONFIG, save_config
from openkb.cli import add_single_file

# Load .env from repo root if present
load_dotenv(Path(__file__).parent / ".env", override=False)

# --- Paths ---
REPO_ROOT = Path(__file__).parent
KB_DIR = REPO_ROOT / "samples" / "kb"
PDF = REPO_ROOT / "samples" / "Blog_RAG.pdf"

LANGUAGE = "en"


def init_kb(kb_dir: Path, language: str) -> None:
    """Create minimal KB structure identical to `openkb init`."""
    (kb_dir / "wiki" / "sources" / "images").mkdir(parents=True, exist_ok=True)
    (kb_dir / "wiki" / "summaries").mkdir(parents=True, exist_ok=True)
    (kb_dir / "wiki" / "concepts").mkdir(parents=True, exist_ok=True)
    (kb_dir / "raw").mkdir(parents=True, exist_ok=True)

    (kb_dir / "wiki" / "AGENTS.md").write_text(AGENTS_MD, encoding="utf-8")
    (kb_dir / "wiki" / "index.md").write_text(
        "# Knowledge Base Index\n\n## Documents\n\n## Concepts\n\n## Explorations\n",
        encoding="utf-8",
    )
    (kb_dir / "wiki" / "log.md").write_text("# Operations Log\n\n", encoding="utf-8")

    openkb_dir = kb_dir / ".openkb"
    openkb_dir.mkdir(exist_ok=True)
    save_config(
        openkb_dir / "config.yaml",
        {
            "model": os.environ.get("OPENAI_MODEL_NAME", DEFAULT_CONFIG["model"]),
            "language": language,
            "pageindex_threshold": DEFAULT_CONFIG["pageindex_threshold"],
            "openai_model_name": os.environ.get("OPENAI_MODEL_NAME", ""),
            "openai_base_url": os.environ.get("OPENAI_BASE_URL", ""),
            "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
            "openai_extra_body": json.loads(os.environ.get("OPENAI_EXTRA_BODY", "{}")),
        },
    )
    (openkb_dir / "hashes.json").write_text(json.dumps({}), encoding="utf-8")
    print(f"KB initialized at {kb_dir}")


def main() -> None:
    if not PDF.exists():
        raise FileNotFoundError(f"PDF not found: {PDF}")

    if not (KB_DIR / ".openkb" / "config.yaml").exists():
        init_kb(KB_DIR, LANGUAGE)

    result = add_single_file(PDF, KB_DIR)
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
