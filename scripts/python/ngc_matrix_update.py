#!/usr/bin/env python3
"""Update the NGC Triton container compatibility matrix conf.

Bash orchestrates the lifecycle; this helper owns the awkward bits that bash is
bad at — fetching the NVIDIA release-notes pages and parsing their HTML tables —
and merges the result into ``scripts/bash/ngc_matrix.conf``.

It is *best-effort*: on any network or parse failure it touches nothing and
exits non-zero, so the caller keeps the existing conf.

Usage:
    python3 ngc_matrix_update.py --conf scripts/bash/ngc_matrix.conf
    python3 ngc_matrix_update.py --conf <path> --print          # stdout, no write
    python3 ngc_matrix_update.py --conf <path> \\
        --index-html idx.html --release-html rel.html           # offline (test)

conf line format (6 space-separated fields):
    <ngc_tag> <min_driver> <tensorrt> <cuda> <python> <size>
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import OrderedDict
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

RELEASE_NOTES_INDEX_URL = (
    "https://docs.nvidia.com/deeplearning/triton-inference-server/"
    "release-notes/index.html"
)
RELEASE_NOTES_BASE_URL = (
    "https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes"
)

# CUDA major.minor, optional update predicate, Linux x86_64 minimum driver.
CUDA_MIN_DRIVER = [
    ("13.2", "update1", "595.58"),
    ("13.2", None, "595.45"),
    ("13.1", "update1", "590.48"),
    ("13.1", None, "590.44"),
    ("13.0", "update2", "580.95"),
    ("13.0", "update1", "580.82"),
    ("13.0", None, "580.65"),
    ("12.9", "update1", "575.57"),
    ("12.9", None, "575.51"),
    ("12.8", "update1", "570.124"),
    ("12.8", None, "570.86"),
    ("12.6", "update3", "560.35"),
    ("12.6", "update2", "560.35"),
    ("12.6", "update1", "560.35"),
    ("12.6", None, "560.28"),
    ("12.5", "update1", "555.42"),
    ("12.5", None, "555.42"),
    ("12.4", "update1", "550.54"),
    ("12.4", None, "550.54"),
    ("12.3", "update1", "545.23"),
    ("12.3", None, "545.23"),
]


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def fetch(url: str, timeout: int = 30) -> str:
    """Fetch a URL as text (raises on failure)."""
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (qwen3tts ngc-matrix)"})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed NVIDIA host
        return resp.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
#  Parsing
# ---------------------------------------------------------------------------

def find_latest_release_url(index_html: str) -> str | None:
    """Return the absolute URL of the newest rel_YY-MM release-notes page."""
    links = re.findall(r'href="([^"]*rel[_-](\d{2})-(\d{2})\.html)[^"]*"', index_html)
    if not links:
        return None
    href = sorted(links, key=lambda it: (int(it[1]), int(it[2])), reverse=True)[0][0]
    if re.match(r"^https?://", href):
        return href
    href = href.split("#", 1)[0]
    return f"{RELEASE_NOTES_BASE_URL}/{href.lstrip('./')}"


def _normalize_cuda(value: str) -> tuple[str, int | None]:
    m = re.search(r"(\d+\.\d+)(?:\.(\d+))?", value)
    if not m:
        return "-", None
    return m.group(1), int(m.group(2) or 0)


def _min_driver_for_cuda(cuda: str, patch: int | None) -> str:
    for major_minor, update, driver in CUDA_MIN_DRIVER:
        if cuda != major_minor:
            continue
        if update == "update3" and (patch or 0) >= 3:
            return driver
        if update == "update2" and (patch or 0) >= 2:
            return driver
        if update == "update1" and (patch or 0) >= 1:
            return driver
        if update is None:
            return driver
    return "-"


def _normalize_version(value: str) -> str:
    m = re.search(r"(\d+(?:\.\d+)+(?:\.post\d+)?(?:\.dev\d+)?)", value)
    return m.group(1) if m else "-"


def _infer_python_version(tag: str) -> str:
    try:
        yy, mm = (int(x) for x in tag.split(".", 1))
    except ValueError:
        return "3.12"
    return "3.12" if (yy, mm) >= (24, 11) else "3.10"


class _TritonReleaseNotesParser(HTMLParser):
    """Collect all HTML tables as lists of rows (each row a list of cells)."""

    def __init__(self) -> None:
        super().__init__()
        self.in_table = self.in_row = self.in_cell = False
        self.current_row: list[str] = []
        self.cell_text = ""
        self.rows: list[list[str]] = []
        self.tables: list[list[list[str]]] = []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.in_table = True
            self.rows = []
        elif tag == "tr" and self.in_table:
            self.in_row = True
            self.current_row = []
        elif tag in ("td", "th") and self.in_row:
            self.in_cell = True
            self.cell_text = ""

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.in_cell:
            self.in_cell = False
            self.current_row.append(self.cell_text.strip())
        elif tag == "tr" and self.in_row:
            self.in_row = False
            if self.current_row:
                self.rows.append(self.current_row)
        elif tag == "table" and self.in_table:
            self.tables.append(self.rows)
            self.in_table = False

    def handle_data(self, data):
        if self.in_cell:
            self.cell_text += data


def scrape_matrix(release_html: str) -> list[list[str]]:
    """Parse the container-versions table into rows of 6 conf fields.

    Returns [] if the expected table is not found.
    """
    parser = _TritonReleaseNotesParser()
    parser.feed(release_html)

    table = None
    for rows in parser.tables:
        if not rows:
            continue
        header = [cell.lower() for cell in rows[0]]
        if (
            any("container version" in cell for cell in header)
            and any("cuda toolkit" in cell for cell in header)
            and any("tensorrt" in cell for cell in header)
        ):
            table = rows
            break
    if not table:
        return []

    out: list[list[str]] = []
    for row in table[1:]:
        if len(row) < 2:
            continue
        m = re.match(r"(\d{2}\.\d{2})$", row[0].strip())
        if not m:
            continue
        ngc_tag = m.group(1)
        cuda, cuda_patch = _normalize_cuda(next((c for c in row if "cuda" in c.lower()), ""))
        tensorrt = _normalize_version(next((c for c in row if "tensorrt" in c.lower()), ""))
        if cuda == "-" or tensorrt == "-":
            continue
        min_driver = _min_driver_for_cuda(cuda, cuda_patch)
        if min_driver == "-":
            continue
        out.append([ngc_tag, min_driver, tensorrt, cuda, _infer_python_version(ngc_tag), "-"])
    return out


# ---------------------------------------------------------------------------
#  Merge
# ---------------------------------------------------------------------------

def parse_conf(path: Path) -> tuple[list[str], "OrderedDict[str, list[str]]"]:
    """Parse a conf file into (comment_lines, {tag: fields}) preserving comments."""
    comments: list[str] = []
    entries: OrderedDict[str, list[str]] = OrderedDict()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            comments.append(line.rstrip())
            continue
        parts = stripped.split()
        if len(parts) >= 4:
            entries[parts[0]] = parts
    return comments, entries


def _tag_sort_key(tag: str) -> tuple[int, int]:
    a, b = tag.split(".")
    return int(a), int(b)


def _pad(fields: list[str], n: int = 6) -> list[str]:
    fields = list(fields)
    while len(fields) < n:
        fields.append("-")
    return fields


def merge_matrix(scraped: list[list[str]], comments: list[str],
                 existing: "OrderedDict[str, list[str]]") -> list[str]:
    """Merge scraped rows over existing entries; return formatted output lines.

    Release-notes data wins for version fields; existing image size is preserved
    when the release-notes table does not publish a size.
    """
    scraped_by_tag = OrderedDict((r[0], r) for r in scraped)
    merged: dict[str, list[str]] = {}
    for tag in set(scraped_by_tag) | set(existing):
        s, e = scraped_by_tag.get(tag), existing.get(tag)
        if s and e:
            result = _pad(s)
            e_padded = _pad(e)
            if result[5] == "-" and e_padded[5] != "-":
                result[5] = e_padded[5]
            merged[tag] = result
        else:
            merged[tag] = _pad(s or e)

    lines: list[str] = []
    for c in comments:
        if "Last updated" in c:
            c = f"#  Last updated: {date.today().isoformat()} (update-matrix)"
        lines.append(c)
    for tag in sorted(merged, key=_tag_sort_key, reverse=True):
        f = merged[tag]
        lines.append(f"{f[0]:6s}  {f[1]:7s}  {f[2]:12s}  {f[3]:5s}  {f[4]:5s}  {f[5]}")
    return lines


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conf", required=True, type=Path, help="Path to ngc_matrix.conf")
    ap.add_argument("--print", action="store_true", dest="to_stdout",
                    help="Print merged conf to stdout instead of writing in place")
    ap.add_argument("--index-html", type=Path, help="Local index HTML (skip fetch; for tests)")
    ap.add_argument("--release-html", type=Path, help="Local release-notes HTML (skip fetch)")
    ap.add_argument("--timeout", type=int, default=30)
    args = ap.parse_args()

    if not args.conf.is_file():
        _log(f"Matrix conf not found: {args.conf}")
        return 1

    # 1) Resolve + fetch the release-notes HTML (or use local files for tests).
    try:
        if args.release_html:
            release_html = args.release_html.read_text(encoding="utf-8", errors="replace")
        else:
            index_html = (
                args.index_html.read_text(encoding="utf-8", errors="replace")
                if args.index_html else fetch(RELEASE_NOTES_INDEX_URL, args.timeout)
            )
            release_url = find_latest_release_url(index_html)
            if not release_url:
                _log("Could not locate latest Triton release-notes page")
                return 1
            release_html = fetch(release_url, args.timeout)
    except Exception as exc:  # noqa: BLE001 - best-effort scraper
        _log(f"Failed to fetch release notes: {exc}")
        return 1

    # 2) Parse the table.
    scraped = scrape_matrix(release_html)
    if not scraped:
        _log("Scraper returned no entries — page structure may have changed")
        return 1
    _log(f"Scraped {len(scraped)} entries from Triton release notes")

    # 3) Merge with the existing conf.
    comments, existing = parse_conf(args.conf)
    merged_lines = merge_matrix(scraped, comments, existing)
    new_count = sum(1 for line in merged_lines if line.strip() and not line.lstrip().startswith("#"))

    if args.to_stdout:
        print("\n".join(merged_lines))
        return 0

    # 4) Write in place with a .bak backup.
    old_count = len(existing)
    backup = args.conf.with_suffix(args.conf.suffix + ".bak")
    backup.write_text(args.conf.read_text(encoding="utf-8"), encoding="utf-8")
    args.conf.write_text("\n".join(merged_lines) + "\n", encoding="utf-8")

    if new_count > old_count:
        _log(f"Matrix updated: {old_count} -> {new_count} entries (+{new_count - old_count} new)")
    elif new_count == old_count:
        _log(f"Matrix up to date ({new_count} entries; versions/sizes may be refreshed)")
    else:
        _log(f"Matrix shrank: {old_count} -> {new_count} (check {backup})")
    _log(f"Backup saved: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
