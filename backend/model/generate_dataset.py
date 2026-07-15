"""
generate_dataset.py

Builds a synthetic-but-realistic training dataset of legitimate and
phishing-style URLs.

NOTE FOR THE README: there is no bundled real-world phishing dataset
in this project (real datasets like PhishTank or the UCI Phishing
Websites dataset are hundreds of MB and require attribution/licensing).
Instead we procedurally generate a large number of URLs that follow
the *patterns* real phishing and legitimate URLs tend to follow, with
randomness mixed in so the model has to learn general patterns rather
than memorize exact strings. This keeps the project fully
self-contained and runnable offline.

For a production system you would swap this out for a real labeled
dataset -- the rest of the pipeline (feature_extractor.py,
train_model.py) doesn't need to change.
"""

import random
import csv
import os
import string

random.seed(42)

REAL_BRANDS = [
    "google", "facebook", "amazon", "microsoft", "apple", "paypal",
    "netflix", "instagram", "twitter", "linkedin", "github", "dropbox",
    "spotify", "adobe", "yahoo", "ebay", "wellsfargo", "chase",
    "bankofamerica", "hsbc",
]

LEGIT_TLDS = [".com", ".org", ".net", ".io", ".co", ".gov", ".edu"]

LEGIT_PATHS = [
    "", "/", "/home", "/about", "/products", "/blog/2024/article",
    "/user/settings", "/docs/getting-started", "/careers",
    "/contact-us", "/support/help-center", "/news/latest",
    "/store/category/electronics", "/api/v1/users",
]

SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "account", "banking",
    "confirm", "password", "signin", "billing", "invoice", "unlock",
    "suspend", "urgent", "wallet", "recover", "alert", "security",
]

SUSPICIOUS_TLDS = [".zip", ".xyz", ".top", ".club", ".work", ".click",
                    ".gq", ".tk", ".ml", ".cf", ".ga", ".men", ".loan"]


def random_string(n, charset=string.ascii_lowercase + string.digits):
    return "".join(random.choice(charset) for _ in range(n))


def make_legit_url():
    brand = random.choice(REAL_BRANDS)
    tld = random.choice(LEGIT_TLDS)
    use_https = random.random() < 0.92
    scheme = "https://" if use_https else "http://"
    use_www = random.random() < 0.5
    sub = "www." if use_www else ""
    path = random.choice(LEGIT_PATHS)
    # occasionally a real subdomain like mail.google.com
    if random.random() < 0.15:
        sub = random.choice(["mail.", "docs.", "shop.", "support.", "api."])
    query = ""
    if random.random() < 0.2:
        query = "?ref=" + random_string(6)
    return f"{scheme}{sub}{brand}{tld}{path}{query}"


def make_phishing_url():
    brand = random.choice(REAL_BRANDS)
    technique = random.choice([
        "typo", "subdomain_trick", "ip", "shortener", "suspicious_tld",
        "long_random", "hyphen_spam", "at_symbol",
    ])
    use_https = random.random() < 0.5  # phishing sites increasingly use https too
    scheme = "https://" if use_https else "http://"
    word = random.choice(SUSPICIOUS_WORDS)

    if technique == "typo":
        # e.g. paypa1.com, arnazon.com
        typo_brand = list(brand)
        idx = random.randrange(len(typo_brand))
        typo_brand[idx] = random.choice("0134589")
        typo_brand = "".join(typo_brand)
        return f"{scheme}{typo_brand}.com/{word}"

    if technique == "subdomain_trick":
        # e.g. paypal.com.verify-account-security.info
        fake_tld = random.choice([".info", ".biz", ".ru", ".cn"])
        return f"{scheme}{brand}.com.{word}-{random_string(5)}{fake_tld}"

    if technique == "ip":
        ip = ".".join(str(random.randint(1, 255)) for _ in range(4))
        return f"{scheme}{ip}/{brand}/{word}.php"

    if technique == "shortener":
        service = random.choice(["bit.ly", "tinyurl.com", "goo.gl", "cutt.ly"])
        return f"{scheme}{service}/{random_string(7)}"

    if technique == "suspicious_tld":
        tld = random.choice(SUSPICIOUS_TLDS)
        return f"{scheme}{brand}-{word}{tld}/{random_string(4)}"

    if technique == "long_random":
        return f"{scheme}{random_string(10)}-{word}-{random_string(8)}.com/{random_string(12)}"

    if technique == "hyphen_spam":
        return f"{scheme}{brand}-{word}-{random_string(4)}-secure-login.com"

    if technique == "at_symbol":
        return f"{scheme}{brand}.com@{random_string(8)}.tk/{word}"

    return f"{scheme}{random_string(10)}.xyz/{word}"


def generate(n_per_class=1500):
    rows = []
    for _ in range(n_per_class):
        rows.append((make_legit_url(), 0))
    for _ in range(n_per_class):
        rows.append((make_phishing_url(), 1))
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dataset.csv")

    rows = generate(1500)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "label"])
        writer.writerows(rows)

    print(f"Generated {len(rows)} rows -> {out_path}")
