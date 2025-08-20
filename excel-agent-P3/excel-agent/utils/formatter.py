def pretty_title(text: str) -> str:
    bar = "═" * (len(text) + 2)
    return f"\n{bar}\n {text}\n{bar}"