"""
Query the indexed KB programmatically.

Usage:
    uv run python run_query.py "What is RAG?"
    uv run python run_query.py  # interactive loop

Env vars (loaded from .env if present):
    OPENAI_MODEL_NAME   OPENAI_BASE_URL   OPENAI_API_KEY   OPENAI_EXTRA_BODY
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=False)

from openkb.config import load_config
from openkb.agent.query import run_query

REPO_ROOT = Path(__file__).parent
KB_DIR = REPO_ROOT / "samples" / "kb"


async def main() -> None:
    if not (KB_DIR / ".openkb" / "config.yaml").exists():
        print(f"KB not initialized at {KB_DIR}. Run run_add.py first.")
        sys.exit(1)

    config = load_config(KB_DIR / ".openkb" / "config.yaml")
    model: str = config.get("model", "gpt-4o-mini")

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        answer = await run_query(question, KB_DIR, model, stream=True)
        print(f"\nAnswer:\n{answer}")
        return

    print("OpenKB query — type question, empty line to quit.\n")
    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        answer = await run_query(question, KB_DIR, model, stream=True)
        print(f"\n{answer}\n")


if __name__ == "__main__":
    asyncio.run(main())
