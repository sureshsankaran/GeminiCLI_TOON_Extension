#!/usr/bin/env python
import json
import logging
import subprocess
import tempfile
from typing import Any

import tiktoken
try:
    from mcp.server.fastmcp import FastMCP
except Exception:
    from fastmcp import FastMCP  # type: ignore

# ================================================================
# LOGGING
# ================================================================
logger = logging.getLogger("toon-mcp")
logging.basicConfig(level=logging.INFO)


# ================================================================
# TOKENIZER (optional but great)
# ================================================================
try:
    tokenizer = tiktoken.get_encoding("o200k_base")
    logger.info("🧮 Loaded GPT o200k_base tokenizer for token savings reporting")
except Exception:
    tokenizer = None
    logger.warning("⚠️ Could not load tokenizer; token savings will be unavailable")


def count_tokens(text: str) -> int:
    if tokenizer is None:
        return -1
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return -1


# ================================================================
# SAFE JSON NORMALIZATION
# ================================================================
def make_json_safe(obj: Any) -> Any:
    """
    Recursively normalize arbitrary Python objects into JSON-safe
    structures (dict/list/str/etc.) so we can dump to JSON cleanly.
    """
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, set):
        return sorted([make_json_safe(v) for v in obj], key=lambda x: str(x))
    if hasattr(obj, "__dict__"):
        return make_json_safe(obj.__dict__)
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


# ================================================================
# TOON CONVERSION (via npx)
# ================================================================
def toon_with_stats(data: Any) -> str:
    """
    Core implementation:
    - Normalizes to JSON-safe
    - Writes temp .json
    - Runs `npx @toon-format/cli` to emit .toon
    - Computes token savings
    - Returns a markdown block with TOON + savings
    """

    safe = make_json_safe(data)
    json_str = json.dumps(safe, indent=2)

    # ------------------------------------------------------------------
    # Run TOON CLI via NPX
    # ------------------------------------------------------------------
    try:
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f_json:
            f_json.write(json_str)
            f_json.flush()
            src = f_json.name
            dst = f_json.name + ".toon"

        cmd = ["toon-format", src, "-o", dst]
        logger.info(f"[TOON] Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"[TOON] CLI failed: {result.stderr}")
            return (
                "```error\n"
                f"TOON CLI failed:\n{result.stderr}\n\n"
                "JSON OUTPUT:\n"
                f"{json_str}\n"
                "```"
            )

        with open(dst, "r") as f:
            toon_str = f.read()

    except Exception as e:
        logger.exception("[TOON] Subprocess error")
        return (
            "```error\n"
            f"TOON subprocess error:\n{e}\n\n"
            "JSON OUTPUT:\n"
            f"{json_str}\n"
            "```"
        )

    # ------------------------------------------------------------------
    # Token savings (FORCED INTO TOOL OUTPUT)
    # ------------------------------------------------------------------
    json_tokens = count_tokens(json_str)
    toon_tokens = count_tokens(toon_str)

    if json_tokens > 0 and toon_tokens > 0:
        reduction = 100 * (1 - (toon_tokens / json_tokens))
        savings_text = (
            f"\n\n# Token Savings\n"
            f"- JSON tokens: {json_tokens}\n"
            f"- TOON tokens: {toon_tokens}\n"
            f"- Saved: {reduction:.1f}%\n"
        )
    else:
        savings_text = "\n\n# Token Savings\n(unavailable)\n"

    # ------------------------------------------------------------------
    # Return TOON + savings info bundled together
    # ------------------------------------------------------------------
    return f"```toon\n{toon_str}\n```{savings_text}"


# ================================================================
# MCP SERVER DEFINITION
# ================================================================
mcp = FastMCP(
    name="toon-json-mcp",
    version="0.1.0",
    description=(
        "Intercepts JSON-like data and converts it to TOON format with "
        "token savings stats, so you can send compact TOON into Gemini."
    ),
)


@mcp.tool()
def to_toon(data: Any) -> str:
    """
    Convert an arbitrary JSON-like object (dict/list/etc.) into TOON text,
    and return TOON plus token savings markdown.

    This is the main tool you will call from Gemini / Gemini CLI.
    """
    return toon_with_stats(data)


@mcp.tool()
def to_toon_from_string(json_text: str) -> str:
    """
    Parse a JSON string and convert it into TOON with token savings.
    Useful if your client has raw JSON text instead of a structured object.
    """
    try:
        parsed = json.loads(json_text)
    except Exception as e:
        raise ValueError(f"Could not parse JSON string: {e}")
    return toon_with_stats(parsed)


if __name__ == "__main__":
    # FastMCP will default to stdio mode if run with no arguments,
    # which is what MCP clients (like Gemini CLI) expect.
    mcp.run()
