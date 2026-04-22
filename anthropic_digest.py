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
        return soup.get_text(" ", strip=True)

    # The article body lives in the .page-wrapper div inside <main>
    body_divs = main.find_all("div", class_="page-wrapper", recursive=False)
    if body_divs:
        return body_divs[0].get_text(" ", strip=True)

    return main.get_text(" ", strip=True)


# ---------------------------------------------------------------------------
# Summarisation with Claude
# ---------------------------------------------------------------------------

SUMMARISE_SYSTEM = """You are writing a self-contained summary of an Anthropic engineering blog post for a weekly digest email. The goal is simple: after reading your summary, the reader should know everything the article covers — well enough that they don't need to open it unless they want to go deeper.

This is NOT a teaser or a highlights reel. Cover the full article, in order. Every major point, result, design decision, and finding should appear in your summary. If the article has five distinct ideas, your summary has five distinct ideas.

Format:
- Write 4–7 sections. Each gets a ## heading and 3–5 sentences of prose.
- Headings should be descriptive and a little interesting — "The 93% Problem" beats "Background".
- Follow the article's own structure and order. Don't rearrange or skip sections.
- Write conversationally. Explain technical concepts in plain terms; use analogies where helpful.
- Be specific: include real numbers, names, system names, and results from the article.
- Target around 600 words total — readable in under 3 minutes.
- Output only the ## sections. No intro sentence, no "In summary", no meta-commentary."""


def summarise_article(client: anthropic.Anthropic, title: str, body: str) -> str:
    """Call Claude to produce a complete, structured summary of an article."""
    if not body.strip():
        return ""

    # Pass the full article — Sonnet handles long context well
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1200,
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
                "content": f'Article title: "{title}"\n\nArticle text:\n{body}',
            }
        ],
    )
    return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# Email builders  (one email per article)
# ---------------------------------------------------------------------------

def _summary_to_html(summary: str) -> str:
    """Convert ## heading + prose markdown to email-safe HTML."""
    html_parts = []
    for block in summary.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("## "):
            heading = block[3:].strip()
            html_parts.append(
                f"<h2 style='margin:28px 0 6px;font-size:17px;font-weight:700;"
                f"color:#1a1a1a;line-height:1.3;'>{heading}</h2>"
            )
        else:
            # May be heading + paragraph joined by a single newline
            lines = block.splitlines()
            out_lines = []
            for line in lines:
                if line.startswith("## "):
                    if out_lines:
                        html_parts.append(
                            f"<p style='margin:0 0 14px;font-size:15px;line-height:1.75;"
                            f"color:#333;'>{' '.join(out_lines)}</p>"
                        )
                        out_lines = []
                    html_parts.append(
                        f"<h2 style='margin:28px 0 6px;font-size:17px;font-weight:700;"
                        f"color:#1a1a1a;line-height:1.3;'>{line[3:].strip()}</h2>"
                    )
                else:
                    out_lines.append(line)
            if out_lines:
                html_parts.append(
                    f"<p style='margin:0 0 14px;font-size:15px;line-height:1.75;"
                    f"color:#333;'>{' '.join(out_lines)}</p>"
                )
    return "\n".join(html_parts)


def _summary_to_text(summary: str) -> str:
    """Strip ## markers for plain-text email."""
    lines = []
    for line in summary.splitlines():
        if line.startswith("## "):
            heading = line[3:].upper()
            lines += ["", heading, "-" * len(heading)]
        else:
            lines.append(line)
    return "\n".join(lines)


def build_article_email(art: dict) -> tuple[str, str, str]:
    """Return (subject, plain_text, html) for a single article email."""
    title = art["title"]
    date_label = art.get("date") or datetime.now().strftime("%B %d, %Y")
    url = art["url"]
    summary = art.get("summary") or art.get("description") or ""

    subject = f"Anthropic Engineering: {title}"

    # --- Plain text ---
    text = "\n".join([
        "ANTHROPIC ENGINEERING",
        "=" * 60,
        "",
        title,
        date_label,
        "",
        _summary_to_text(summary),
        "",
        "-" * 60,
        f"Read the full article: {url}",
        "View all: https://www.anthropic.com/engineering",
    ])

    # --- HTML ---
    body_html = _summary_to_html(summary)
    html = f"""<!DOCTYPE html>
<html>
<body style='font-family:Georgia,serif;max-width:620px;margin:0 auto;padding:40px 24px;background:#fff;color:#1a1a1a;'>
  <table width='100%' cellpadding='0' cellspacing='0'>
    <tr><td>

      <!-- Header -->
      <p style='margin:0 0 20px;font-size:11px;letter-spacing:.1em;text-transform:uppercase;
                color:#d97757;font-weight:700;font-family:system-ui,sans-serif;'>
        Anthropic Engineering
      </p>

      <!-- Title -->
      <h1 style='margin:0 0 8px;font-size:26px;line-height:1.25;font-weight:700;color:#1a1a1a;'>
        <a href='{url}' style='color:#1a1a1a;text-decoration:none;'>{title}</a>
      </h1>
      <p style='margin:0 0 28px;font-size:13px;color:#999;font-family:system-ui,sans-serif;'>
        {date_label}
      </p>

      <hr style='border:none;border-top:2px solid #1a1a1a;margin-bottom:28px;'>

      <!-- Summary body -->
      {body_html}

      <!-- CTA -->
      <div style='margin-top:36px;padding-top:24px;border-top:1px solid #eee;'>
        <a href='{url}'
           style='display:inline-block;padding:11px 22px;background:#d97757;color:#fff;
                  font-size:14px;font-weight:600;text-decoration:none;border-radius:5px;
                  font-family:system-ui,sans-serif;'>
          Read the full article &rarr;
        </a>
      </div>

      <!-- Footer -->
      <p style='margin-top:32px;font-size:11px;color:#bbb;font-family:system-ui,sans-serif;'>
        <a href='https://www.anthropic.com/engineering' style='color:#bbb;'>
          anthropic.com/engineering
        </a>
      </p>

    </td></tr>
  </table>
</body>
</html>"""

    return subject, text, html


# ---------------------------------------------------------------------------
# Email sender
# ---------------------------------------------------------------------------

def send_email(config, subject: str, text_body: str, html_body: str):
    gmail_user = config["gmail"]["user"].strip()
    gmail_pass = config["gmail"]["app_password"].strip()

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"Anthropic Engineering <{gmail_user}>"
    msg["To"] = gmail_user

    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.ehlo()
        server.starttls()
        server.login(gmail_user, gmail_pass)
        server.sendmail(gmail_user, gmail_user, msg.as_string())

    print(f"  Sent: {subject}")


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

    print(f"Found {len(new_articles)} new article(s) — fetching and summarising each...")
    for art in new_articles:
        print(f"  → {art['title']}")
        body = fetch_article_text(art["url"])
        art["summary"] = summarise_article(claude, art["title"], body)
        subject, text_body, html_body = build_article_email(art)
        send_email(config, subject, text_body, html_body)

    seen.update(a["url"] for a in articles)
    save_seen(seen)
    print("Done.")


if __name__ == "__main__":
    main()
