#!/usr/bin/env python3
"""
Anthropic Engineering Blog Digest
Scrapes https://www.anthropic.com/engineering for new articles, summarizes each
one with Claude, and emails the digest via Gmail SMTP.
"""

import json
import os
import smtplib
import configparser
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import anthropic
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.anthropic.com"
BLOG_URL = f"{BASE_URL}/engineering"

SCRIPT_DIR = Path(__file__).parent
SEEN_FILE = SCRIPT_DIR / "seen_articles.json"
CONFIG_FILE = SCRIPT_DIR / "config.ini"

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# Config / state helpers
# ---------------------------------------------------------------------------

def load_config():
    # Environment variables take precedence — used by GitHub Actions
    if os.environ.get("GMAIL_USER"):
        config = configparser.ConfigParser()
        config["gmail"] = {
            "user": os.environ["GMAIL_USER"],
            "app_password": os.environ.get("GMAIL_APP_PASSWORD", ""),
        }
        config["anthropic"] = {
            "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        }
        return config

    # Fall back to config.ini for local use
    config = configparser.ConfigParser()
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Config file not found: {CONFIG_FILE}\n"
            "Copy config.ini.example to config.ini and fill in your credentials."
        )
    config.read(CONFIG_FILE)
    return config


def load_seen():
    if SEEN_FILE.exists():
        with open(SEEN_FILE) as f:
            return set(json.load(f))
    return set()


def save_seen(seen):
    with open(SEEN_FILE, "w") as f:
        json.dump(sorted(seen), f, indent=2)


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

def fetch_articles():
    """Return list of article dicts from the engineering index page."""
    resp = requests.get(BLOG_URL, headers=REQUEST_HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    articles = []

    # Strategy 1: __NEXT_DATA__ JSON (most reliable when present)
    next_data_tag = soup.find("script", id="__NEXT_DATA__")
    if next_data_tag:
        try:
            data = json.loads(next_data_tag.string)
            props = data.get("props", {}).get("pageProps", {})
            posts = (
                props.get("posts")
                or props.get("articles")
                or props.get("engineeringPosts")
                or props.get("items")
            )
            if posts and isinstance(posts, list):
                for post in posts:
                    slug = post.get("slug", post.get("url", ""))
                    if slug and not slug.startswith("http"):
                        slug = f"/engineering/{slug.lstrip('/')}"
                    articles.append({
                        "title": post.get("title", "").strip(),
                        "url": BASE_URL + slug if slug else "",
                        "date": post.get("date", post.get("publishedAt", "")),
                        "description": post.get("description", post.get("summary", "")),
                        "summary": "",
                    })
                if articles:
                    return [a for a in articles if a["url"] and a["title"]]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # Strategy 2: Parse <article> cards directly
    for card in soup.find_all("article"):
        link = card.find("a", href=True)
        if not link:
            continue
        href = link["href"]
        if not href.startswith("/engineering/") or href.rstrip("/") == "/engineering":
            continue

        title_el = card.find(["h2", "h3", "h4"])
        desc_el = card.find("p")
        date_el = card.find(
            lambda t: t.name in ("div", "span", "time")
            and any("__date" in c for c in (t.get("class") or []))
        )

        title = title_el.get_text(strip=True) if title_el else ""
        description = desc_el.get_text(strip=True) if desc_el else ""
        date = date_el.get_text(strip=True) if date_el else ""

        if title:
            articles.append({
                "title": title,
                "url": BASE_URL + href,
                "date": date,
                "description": description,
                "summary": "",
            })

    return articles


def fetch_article_text(url: str) -> str:
    """Fetch an article page and return the main body text."""
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"    Warning: could not fetch {url}: {e}")
        return ""

    soup = BeautifulSoup(resp.text, "lxml")
    main = soup.find("main")
    if not main:
        return soup.get_text(" ", strip=True)[:8000]

    # The article body lives in the second .page-wrapper div inside <main>
    body_divs = main.find_all("div", class_="page-wrapper", recursive=False)
    if body_divs:
        return body_divs[0].get_text(" ", strip=True)

    return main.get_text(" ", strip=True)


# ---------------------------------------------------------------------------
# Summarisation with Claude
# ---------------------------------------------------------------------------

SUMMARISE_SYSTEM = (
    "You are a sharp, opinionated technical writer crafting the summary section of a "
    "blog-style weekly digest email. Write as if you're talking directly to a curious "
    "engineer — conversational but substantive. "
    "Structure your response as 2–3 short paragraphs: open with a hook that captures "
    "what makes this piece interesting, then walk through the key technical idea or "
    "finding (be specific — cite real numbers, techniques, or results from the article), "
    "and close with a sentence on why it matters or what to watch next. "
    "No bullet points, no headers, no markdown. Just clean readable prose that feels "
    "like it was written by someone who actually read and enjoyed the article."
)


def summarise_article(client: anthropic.Anthropic, title: str, body: str) -> str:
    """Call Claude to produce a blog-style summary of an article."""
    if not body.strip():
        return ""

    trimmed = body[:6000]

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        system=[
            {
                "type": "text",
                "text": SUMMARISE_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": f'Article title: "{title}"\n\nArticle text:\n{trimmed}',
            }
        ],
    )
    return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# Email builders
# ---------------------------------------------------------------------------

def build_text_body(new_articles):
    lines = [
        "ANTHROPIC ENGINEERING DIGEST",
        datetime.now().strftime("%A, %B %d, %Y"),
        "=" * 60,
        "",
    ]

    # Featured article
    featured = new_articles[0]
    lines.append("[ FEATURED ]")
    lines.append(featured["title"])
    if featured.get("date"):
        lines.append(featured["date"])
    lines.append("")
    if featured.get("summary"):
        lines.append(featured["summary"])
    lines.append("")
    lines.append(featured["url"])

    # Rest
    if len(new_articles) > 1:
        lines += ["", "-" * 60, "ALSO NEW", ""]
        for art in new_articles[1:]:
            lines.append(f"* {art['title']}")
            if art.get("date"):
                lines.append(f"  {art['date']}")
            if art.get("summary"):
                lines.append(f"  {art['summary']}")
            lines.append(f"  {art['url']}")
            lines.append("")

    lines += ["", "---", "View all: https://www.anthropic.com/engineering"]
    return "\n".join(lines)


def _article_li(art: dict) -> str:
    """Render a standard (non-featured) article list item."""
    summary_text = art.get("summary") or art.get("description") or ""
    summary_html = (
        f"<p style='margin:8px 0 0;color:#444;font-size:14px;line-height:1.6;'>"
        f"{summary_text}</p>"
    ) if summary_text else ""
    date_html = (
        f"<span style='font-size:12px;color:#888;display:block;margin-top:4px;'>"
        f"{art['date']}</span>"
    ) if art.get("date") else ""
    return f"""
    <li style='margin-bottom:28px;list-style:none;padding-left:14px;border-left:3px solid #d97757;'>
      <a href='{art["url"]}' style='font-size:16px;font-weight:600;color:#1a1a1a;text-decoration:none;'>
        {art["title"]}
      </a>
      {date_html}
      {summary_html}
      <p style='margin:8px 0 0;'>
        <a href='{art["url"]}' style='font-size:12px;color:#d97757;text-decoration:none;'>
          Read article &rarr;
        </a>
      </p>
    </li>"""


def _featured_hero(art: dict) -> str:
    """Render the top article as a large hero block."""
    summary_text = art.get("summary") or art.get("description") or ""
    # Split into paragraphs for the hero so it reads like a blog excerpt
    paras = [p.strip() for p in summary_text.split("\n\n") if p.strip()]
    paras_html = "".join(
        f"<p style='margin:0 0 12px;font-size:15px;line-height:1.7;color:#333;'>{p}</p>"
        for p in paras
    ) if paras else (
        f"<p style='margin:0 0 12px;font-size:15px;line-height:1.7;color:#333;'>{summary_text}</p>"
        if summary_text else ""
    )
    date_html = (
        f"<span style='font-size:12px;color:#888;'>{art['date']}</span>"
    ) if art.get("date") else ""

    return f"""
    <div style='background:#fdf6f2;border-radius:8px;padding:24px 24px 20px;margin-bottom:32px;'>
      <p style='margin:0 0 6px;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#d97757;font-weight:700;'>
        Featured
      </p>
      <h2 style='margin:0 0 6px;font-size:20px;line-height:1.3;'>
        <a href='{art["url"]}' style='color:#1a1a1a;text-decoration:none;'>{art["title"]}</a>
      </h2>
      {date_html}
      <div style='margin-top:14px;'>
        {paras_html}
      </div>
      <a href='{art["url"]}' style='display:inline-block;margin-top:4px;font-size:13px;font-weight:600;color:#d97757;text-decoration:none;'>
        Read article &rarr;
      </a>
    </div>"""


def build_html_body(new_articles):
    date_label = datetime.now().strftime("%A, %B %d, %Y")
    count_label = f"{len(new_articles)} new article{'s' if len(new_articles) != 1 else ''}"

    hero_html = _featured_hero(new_articles[0])

    rest_html = ""
    if len(new_articles) > 1:
        items = "".join(_article_li(a) for a in new_articles[1:])
        rest_html = f"""
        <h3 style='font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:#888;margin:0 0 20px;'>
          Also new
        </h3>
        <ul style='padding:0;margin:0;'>{items}</ul>"""

    return f"""<!DOCTYPE html>
<html>
<body style='font-family:system-ui,-apple-system,sans-serif;max-width:640px;margin:0 auto;padding:32px 24px;background:#fff;color:#1a1a1a;'>
  <table width='100%' cellpadding='0' cellspacing='0'>
    <tr>
      <td>
        <p style='margin:0 0 2px;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:#d97757;font-weight:600;'>
          Anthropic Engineering Digest
        </p>
        <p style='margin:0 0 24px;font-size:13px;color:#aaa;'>{date_label} &middot; {count_label}</p>
        <hr style='border:none;border-top:1px solid #eee;margin-bottom:24px;'>
        {hero_html}
        {rest_html}
        <hr style='border:none;border-top:1px solid #eee;margin-top:32px;margin-bottom:16px;'>
        <p style='font-size:12px;color:#aaa;margin:0;'>
          <a href='https://www.anthropic.com/engineering' style='color:#aaa;'>
            View all articles on anthropic.com/engineering
          </a>
        </p>
      </td>
    </tr>
  </table>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Email sender
# ---------------------------------------------------------------------------

def send_email(config, subject, text_body, html_body):
    gmail_user = config["gmail"]["user"].strip()
    gmail_pass = config["gmail"]["app_password"].strip()
    to_addr = gmail_user

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"Anthropic Digest <{gmail_user}>"
    msg["To"] = to_addr

    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.ehlo()
        server.starttls()
        server.login(gmail_user, gmail_pass)
        server.sendmail(gmail_user, to_addr, msg.as_string())

    print(f"  Email sent to {to_addr}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = load_config()
    api_key = config["anthropic"]["api_key"].strip()
    claude = anthropic.Anthropic(api_key=api_key)

    seen = load_seen()

    print(f"[{datetime.now():%Y-%m-%d %H:%M}] Fetching {BLOG_URL} ...")
    articles = fetch_articles()

    if not articles:
        print("No articles found — page structure may have changed.")
        return

    new_articles = [a for a in articles if a["url"] not in seen]

    if not new_articles:
        print(f"No new articles. ({len(articles)} total on page, all already seen.)")
        seen.update(a["url"] for a in articles)
        save_seen(seen)
        return

    print(f"Found {len(new_articles)} new article(s) — fetching content and summarising...")
    for art in new_articles:
        print(f"  Summarising: {art['title']}")
        body = fetch_article_text(art["url"])
        art["summary"] = summarise_article(claude, art["title"], body)

    subject = (
        f"Anthropic Engineering: {len(new_articles)} new "
        f"article{'s' if len(new_articles) != 1 else ''}"
    )
    text_body = build_text_body(new_articles)
    html_body = build_html_body(new_articles)

    send_email(config, subject, text_body, html_body)

    seen.update(a["url"] for a in articles)
    save_seen(seen)
    print("Done.")


if __name__ == "__main__":
    main()
