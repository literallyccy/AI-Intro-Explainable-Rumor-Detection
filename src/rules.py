import re

DEBUNK_PATTERNS = [
    r"\b(police|officials?|authorit(?:y|ies)|government|ministry|agency|experts?|doctor|doctors|scientists?)\b.{0,80}\b(false|fake|untrue|misleading|debunked|denied|fabricated|not true|no evidence)\b",
    r"\b(false|fake|untrue|misleading|debunked|denied|fabricated|not true)\b.{0,80}\b(rumou?rs?|claim|claims|post|story|message|news)\b",
    r"\bnot\s+to\s+spread\s+rumou?rs?\b",
    r"\bconfirmed\s+that\b.{0,120}\bwas\s+false\b",
]


def detect_debunking(text):
    text_l = str(text).lower()
    for pattern in DEBUNK_PATTERNS:
        if re.search(pattern, text_l):
            return True, pattern
    return False, ""
