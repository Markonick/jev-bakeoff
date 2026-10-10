"""Robust JSON extraction from LLM text output.

Models occasionally wrap JSON in prose or code fences, or emit a stray leading
line. `parse_json_object` tolerates those: it strips code fences and, failing a
direct parse, falls back to the outermost `{...}` span. Raises ValueError when no
JSON object can be recovered so callers can retry or fail gracefully.
"""

import json
from typing import Any, cast


def parse_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        # Drop the opening fence line (``` or ```json) and the trailing fence.
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        return cast(dict[str, Any], json.loads(text))
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost braces — handles leading/trailing prose.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return cast(dict[str, Any], json.loads(text[start : end + 1]))

    raise ValueError("no JSON object found in model output")
