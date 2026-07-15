"""
feature_extractor.py

Turns a raw URL string into a numeric feature vector that the
machine learning model can understand.

The features below are all "lexical" features -- meaning they are
computed purely from the text of the URL itself (no need to visit
the site). This is a common, lightweight approach used in real
phishing-detection research because it is fast and doesn't require
network access at prediction time.
"""

import re
import math
from urllib.parse import urlparse

# A short list of words that show up disproportionately often in
# phishing URLs (login pages, "verify your account" scams, etc.)
SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "account", "banking",
    "confirm", "password", "signin", "sign-in", "webscr", "ebayisapi",
    "paypal", "billing", "invoice", "unlock", "suspend", "urgent",
    "click", "免费", "free", "gift", "bonus", "prize", "wallet",
    "recover", "support", "helpdesk", "security", "alert",
]

# Common URL-shortening services. Shortened links hide the real
# destination, which is a classic phishing trick.
SHORTENING_SERVICES = [
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "shorte.st", "cutt.ly", "rebrand.ly",
]

SUSPICIOUS_TLDS = [
    ".zip", ".xyz", ".top", ".club", ".work", ".click", ".link",
    ".gq", ".tk", ".ml", ".cf", ".ga", ".men", ".loan", ".download",
]

IP_PATTERN = re.compile(
    r"^(https?://)?(\d{1,3}\.){3}\d{1,3}"
)

FEATURE_NAMES = [
    "url_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_digits",
    "num_at_symbols",
    "num_question_marks",
    "num_equal_signs",
    "num_subdomains",
    "has_ip_address",
    "has_https",
    "has_port",
    "has_suspicious_word",
    "has_shortening_service",
    "has_suspicious_tld",
    "domain_length",
    "path_length",
    "digit_letter_ratio",
    "shannon_entropy",
]


def _shannon_entropy(s: str) -> float:
    """Higher entropy roughly means a more 'random looking' string,
    which is common in auto-generated phishing domains."""
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def is_valid_url(url: str) -> tuple:
    """
    Basic structural sanity check for a URL, run BEFORE feature
    extraction / prediction.

    This does not check whether the site actually exists (no network
    calls) -- it only checks that the text looks like a real URL
    (has a domain-like structure), so the model isn't asked to judge
    garbage input like "v", "hello", or an empty string.

    Returns (True, "") if valid, or (False, "<reason>") if not.
    """
    url = (url or "").strip()

    if not url:
        return False, "Please enter a URL."

    if len(url) < 4:
        return False, "That doesn't look like a valid URL (too short)."

    if len(url) > 2048:
        return False, "That URL is too long to be valid."

    if " " in url:
        return False, "URLs can't contain spaces."

    # Strip a scheme if present, so we can inspect just the domain part
    candidate = url
    if re.match(r"^https?://", candidate, re.IGNORECASE):
        candidate = re.sub(r"^https?://", "", candidate, flags=re.IGNORECASE)
    elif "://" in candidate:
        # Some other scheme (ftp://, javascript:, etc.) -- reject
        return False, "Only http:// and https:// URLs are supported."

    # Take just the domain portion (before the first slash/query/hash)
    domain = re.split(r"[/?#]", candidate, maxsplit=1)[0]
    domain = domain.split("@")[-1]  # drop userinfo like user@ if present
    domain = domain.split(":")[0]   # drop a port if present

    if not domain:
        return False, "That doesn't look like a valid URL (missing domain)."

    # A real domain must either be an IP address, or contain at least
    # one dot with real characters on both sides (e.g. "example.com").
    if IP_PATTERN.match(domain) or re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain):
        return True, ""

    if "." not in domain:
        return False, (
            "That doesn't look like a valid domain (expected something "
            "like \"example.com\")."
        )

    labels = domain.split(".")
    if any(len(label) == 0 for label in labels):
        return False, "That doesn't look like a valid domain."

    if not re.match(r"^[a-zA-Z0-9.\-]+$", domain):
        return False, "That domain contains characters that aren't valid in a URL."

    # Last label (the TLD) should be alphabetic and at least 2 characters
    if not re.match(r"^[a-zA-Z]{2,}$", labels[-1]):
        return False, "That doesn't look like a valid domain ending (e.g. \".com\")."

    return True, ""


def extract_features(url: str) -> list:
    """Given a raw URL string, return a list of numeric features in
    the exact order of FEATURE_NAMES."""

    url = url.strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        # Assume http(s) was omitted, still parse what we can
        parse_target = "http://" + url
    else:
        parse_target = url

    parsed = urlparse(parse_target)
    domain = parsed.netloc.lower()
    path = parsed.path or ""
    lower_url = url.lower()

    num_subdomains = max(domain.count(".") - 1, 0)

    features = [
        len(url),
        url.count("."),
        url.count("-"),
        url.count("_"),
        url.count("/"),
        sum(c.isdigit() for c in url),
        url.count("@"),
        url.count("?"),
        url.count("="),
        num_subdomains,
        1 if IP_PATTERN.match(url) else 0,
        1 if lower_url.startswith("https://") else 0,
        1 if ":" in domain else 0,
        1 if any(word in lower_url for word in SUSPICIOUS_WORDS) else 0,
        1 if any(s in lower_url for s in SHORTENING_SERVICES) else 0,
        1 if any(lower_url.endswith(t) or (t + "/") in lower_url for t in SUSPICIOUS_TLDS) else 0,
        len(domain),
        len(path),
        (sum(c.isdigit() for c in url) / max(sum(c.isalpha() for c in url), 1)),
        _shannon_entropy(url),
    ]

    return features
