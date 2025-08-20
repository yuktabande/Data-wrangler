import os
from pathlib import Path

def ensure_output_dir() -> None:
    Path("output").mkdir(parents=True, exist_ok=True)

def resolve_input_path(cli_arg: str | None) -> str:
    """
    If user passes a path via CLI, use it.
    Else default to: data/sample.xlsx OR input/Input+metadata.xlsx if you prefer.
    """
    candidates = []
    if cli_arg:
        candidates.append(cli_arg)
    # prefer your existing location first:
    candidates.append("input/Input+metadata.xlsx")
    # alt demo path:
    candidates.append("data/sample.xlsx")

    for p in candidates:
        if os.path.isfile(p):
            return p

    raise FileNotFoundError(
        "Could not locate an input Excel file. "
        "Pass a path as an argument or place a file at input/Input+metadata.xlsx"
    )

def write_text_file(path: str, content: str) -> None:
    ensure_output_dir()
    with open(path, "w", encoding="utf-8") as f:
        f.write(content if content.endswith("\n") else content + "\n")