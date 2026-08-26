"""
Timestamps in AST (Atlantic Standard Time, UTC-4, no DST) — platform rule:
every VICARIUS timestamp is AST 24h, never UTC or 12h.
"""

from datetime import datetime, timezone, timedelta

AST = timezone(timedelta(hours=-4))


def now_ast():
    """'YYYY-MM-DD HH:MM:SS' in AST 24h."""
    return datetime.now(AST).strftime('%Y-%m-%d %H:%M:%S')


def now_ast_iso():
    """ISO-8601 with the -04:00 offset, AST."""
    return datetime.now(AST).replace(microsecond=0).isoformat()
