"""Helper konsol Windows (§11.5): paksa stdout UTF-8 agar literal seperti ≈ → Δ − ✅
tidak crash dengan `UnicodeEncodeError` di codepage cp1252.

Pakai di awal setiap skrip CLI: `from tmt2.console import ensure_utf8; ensure_utf8()`.
Atau set `PYTHONUTF8=1` di runner.
"""


def ensure_utf8():
    try:
        import sys
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
