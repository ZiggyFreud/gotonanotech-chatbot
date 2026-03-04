import re
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from urllib.parse import urlparse

SITEMAP_URL = "https://gotonanotech.com/sitemap_index.xml"
OUTPUT_FILE = "data/website_text.txt"
MAX_PAGES = 500
TIMEOUT = 25

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; WPTextScraper/1.0)"
}

WP_CONTENT_SELECTORS = [
    ".entry-content",
    "article",
    "main",
    ".post-content",
    ".page-content",
    "#content",
    "#primary",
]

REMOVE_SELECTORS = [
    "header", "footer", "nav", "aside",
    ".site-header", ".site-footer", ".menu", ".navigation", ".nav",
    ".sidebar", ".widget", ".widget-area",
    ".comments-area", "#comments", ".comment-respond",
    ".sharedaddy", ".jp-relatedposts",
    ".breadcrumb", ".breadcrumbs",
    ".cookie", ".cookie-banner", ".gdpr",
]

BAD_LINE_PATTERNS = [
    r"^leave a comment$",
    r"^reply$",
    r"^uncategorized$",
    r"^by$",
    r"^by\s+.+$",
    r"^contact us$",
    r".+@.+\..+",                 # emails
    r"^search$",
]

BLOCK_URL_CONTAINS = [
    "/tag/",
    "/category/",
    "/author/",
    "/page/",                      # pagination like /page/2/
    "/wp-json/",
    "/feed/",
]

def fetch(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text

def parse_sitemap_urls(sitemap_url: str) -> list[str]:
    xml_text = fetch(sitemap_url)
    root = ET.fromstring(xml_text)
    ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    # Sitemap index: <sitemap><loc>...</loc></sitemap>
    sitemap_locs = [
        loc.text.strip()
        for loc in root.findall(".//ns:sitemap/ns:loc", ns)
        if loc.text
    ]
    if sitemap_locs:
        urls: list[str] = []
        for sm in sitemap_locs:
            urls.extend(parse_sitemap_urls(sm))
        return urls

    # Regular sitemap: <url><loc>...</loc></url>
    return [
        loc.text.strip()
        for loc in root.findall(".//ns:url/ns:loc", ns)
        if loc.text
    ]

def clean_lines(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cleaned: list[str] = []

    for ln in lines:
        lower = ln.lower()

        # Remove boilerplate lines
        drop = False
        for pat in BAD_LINE_PATTERNS:
            if re.fullmatch(pat, lower):
                drop = True
                break
        if drop:
            continue

        cleaned.append(ln)

    return "\n".join(cleaned)

def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts and styles
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    # Remove common layout / plugin blocks
    for sel in REMOVE_SELECTORS:
        for node in soup.select(sel):
            node.decompose()

    # Pick best content container by length
    best_node = None
    best_len = 0

    for sel in WP_CONTENT_SELECTORS:
        for node in soup.select(sel):
            txt = node.get_text(separator="\n", strip=True)
            if len(txt) > best_len:
                best_len = len(txt)
                best_node = node

    target = best_node if best_node else (soup.body or soup)
    raw_text = target.get_text(separator="\n", strip=True)

    # Normalize spacing
    raw_text = raw_text.replace("\r", "\n")
    raw_text = re.sub(r"\n{3,}", "\n\n", raw_text)

    return clean_lines(raw_text)

def main():
    print("Starting full scrape from sitemap...")

    urls = parse_sitemap_urls(SITEMAP_URL)

    # Keep only same domain as sitemap
    domain = urlparse(SITEMAP_URL).netloc
    urls = [u for u in urls if urlparse(u).netloc == domain]

    # Remove unwanted URL types
    urls = [u for u in urls if not any(bad in u for bad in BLOCK_URL_CONTAINS)]

    # Cap pages
    urls = urls[:MAX_PAGES]

    print(f"Found {len(urls)} URLs to scrape")

    out_chunks: list[str] = []

    for i, url in enumerate(urls, start=1):
        try:
            print(f"[{i}/{len(urls)}] Scraping: {url}")
            html = fetch(url)
            text = extract_main_text(html)

            # Skip thin pages
            if len(text) < 400:
                continue

            out_chunks.append(f"\n\n===== {url} =====\n\n{text}")

        except Exception as e:
            print(f"Failed: {url} | {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(out_chunks))

    print(f"Done. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()