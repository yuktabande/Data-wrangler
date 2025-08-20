# excel-agent/agent/suggestions.py
import os
import json
import textwrap
import google.generativeai as genai
from config import GOOGLE_API_KEY

# Configure Gemini once per process
def _configure():
    api_key = GOOGLE_API_KEY
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to .env, e.g.\n"
            'GOOGLE_API_KEY="YOUR_KEY"'
        )
    genai.configure(api_key=api_key)

def _truncate_for_prompt(obj: dict, max_chars: int = 120_000) -> str:
    """
    Safely stringify JSON and cap length so prompts don't explode.
    Converts non-JSON types (e.g., pandas Timestamps) via default=str.
    """
    s = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    if len(s) <= max_chars:
        return s
    # soft truncate on line boundary
    s = s[:max_chars]
    last_nl = s.rfind("\n")
    return s[:last_nl] + "\n... (truncated for prompt) ..."

def generate_suggestions(sheet_summaries: dict, sheet_samples: dict | None = None) -> str:
    _configure()
    model = genai.GenerativeModel("gemini-1.5-flash")

    summaries_str = _truncate_for_prompt(sheet_summaries)
    samples_str = _truncate_for_prompt(sheet_samples or {})

    system_instructions = textwrap.dedent("""
        You are a helpful data wrangling assistant.
        Goal: Given Excel sheet summaries and tiny samples, propose 5–8 meaningful, actionable operations.
        Favor concrete actions like joins, pivots, filters, cleaning, deduping, outlier checks, and visualizations.

        Output requirements:
        - Use a concise, numbered list.
        - Each item should name the sheet(s) and column(s) involved.
        - Prefer specific joins (e.g., "Join Sales & Returns on Order ID").
        - Include at least one data quality check (missing values / duplicates).
        - Include at least one visualization suggestion if there is a time/numeric column.
    """)

    user_content = f"""
== Sheet Summaries ==
{summaries_str}

== Sheet Samples (first rows) ==
{samples_str}
"""

    prompt = system_instructions.strip() + "\n\n" + user_content.strip()

    resp = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.4,
            "top_p": 0.9,
            "max_output_tokens": 700,
        }
    )

    # Primary path
    if getattr(resp, "text", None):
        return resp.text.strip()

    # Fallback: concatenate candidate parts, if present
    if hasattr(resp, "candidates") and resp.candidates:
        for cand in resp.candidates:
            if cand.content and getattr(cand.content, "parts", None):
                joined = "".join(getattr(p, "text", "") for p in cand.content.parts)
                if joined.strip():
                    return joined.strip()

    return "⚠️ No suggestions returned by Gemini."