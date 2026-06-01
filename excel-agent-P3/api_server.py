#!/usr/bin/env python3
"""Small HTTP bridge for the Excel agent frontend.

Run from the repo root with:
    python3 excel-agent-P3/api_server.py
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


APP_DIR = Path(__file__).resolve().parent / "excel-agent"
DATA_DIR = APP_DIR / "data"
OUTPUT_DIR = APP_DIR / "output"
INPUT_FILE = DATA_DIR / "input.xlsx"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _run_in_app_dir(fn):
    previous = Path.cwd()
    os.chdir(APP_DIR)
    try:
        return fn()
    finally:
        os.chdir(previous)


def _summarize_input(include_suggestions: bool = False) -> dict[str, Any]:
    def work() -> dict[str, Any]:
        from agent.loader import sample_sheets, summarize_excel

        summary = summarize_excel(str(INPUT_FILE))
        samples = sample_sheets(str(INPUT_FILE), n=5)
        payload: dict[str, Any] = {"summary": summary, "samples": samples}

        OUTPUT_DIR.mkdir(exist_ok=True)
        (OUTPUT_DIR / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )

        if include_suggestions:
            try:
                from agent.suggestions import generate_suggestions

                suggestions = generate_suggestions(summary, samples)
            except Exception as exc:  # Keep analysis usable without Gemini.
                suggestions = f"Could not fetch AI suggestions: {exc}"
            (OUTPUT_DIR / "suggestions.txt").write_text(suggestions, encoding="utf-8")
            payload["suggestions"] = suggestions

        return payload

    return _run_in_app_dir(work)


def _execute_instruction(instruction: str) -> dict[str, Any]:
    def work() -> dict[str, Any]:
        from agent.executor import execute_natural_language

        return execute_natural_language(str(INPUT_FILE), instruction)

    return _run_in_app_dir(work)


class Handler(BaseHTTPRequestHandler):
    server_version = "ExcelAgentAPI/1.0"

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(_json_safe(payload), default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Filename")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def do_OPTIONS(self) -> None:
        self._send_json({"ok": True})

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/api/health", "/api/status"}:
            self._send_json(
                {
                    "ok": True,
                    "inputFile": str(INPUT_FILE),
                    "inputExists": INPUT_FILE.exists(),
                    "outputDir": str(OUTPUT_DIR),
                }
            )
            return

        self._send_json({"ok": False, "error": "Not found"}, status=404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/upload":
                length = int(self.headers.get("Content-Length", "0"))
                if length == 0:
                    self._send_json({"ok": False, "error": "No file body received"}, status=400)
                    return

                filename = self.headers.get("X-Filename", "input.xlsx")
                if not filename.lower().endswith((".xlsx", ".xls")):
                    self._send_json({"ok": False, "error": "Upload an Excel file"}, status=400)
                    return

                DATA_DIR.mkdir(parents=True, exist_ok=True)
                INPUT_FILE.write_bytes(self.rfile.read(length))
                analysis = _summarize_input(include_suggestions=True)
                self._send_json({"ok": True, "filename": filename, **analysis})
                return

            if path == "/api/analyze":
                if not INPUT_FILE.exists():
                    self._send_json({"ok": False, "error": "No input.xlsx found"}, status=404)
                    return
                analysis = _summarize_input(include_suggestions=True)
                self._send_json({"ok": True, **analysis})
                return

            if path == "/api/execute":
                payload = self._read_json()
                instruction = str(payload.get("instruction", "")).strip()
                if not instruction:
                    self._send_json({"ok": False, "error": "Instruction is required"}, status=400)
                    return
                if not INPUT_FILE.exists():
                    self._send_json({"ok": False, "error": "No input.xlsx found"}, status=404)
                    return

                result = _execute_instruction(instruction)
                self._send_json({"ok": True, **result})
                return

            self._send_json({"ok": False, "error": "Not found"}, status=404)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=500)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    host = os.getenv("EXCEL_AGENT_HOST", "127.0.0.1")
    port = int(os.getenv("EXCEL_AGENT_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Excel Agent API running at http://{host}:{port}")
    print(f"Using input file: {INPUT_FILE}")
    server.serve_forever()


if __name__ == "__main__":
    main()
