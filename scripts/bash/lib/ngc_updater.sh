#!/bin/bash
# ===========================================================================
#  ngc_updater.sh — Fetch and update NGC compatibility matrix
#
#  Scrapes the official NVIDIA Triton release notes and merges new entries
#  into the local ngc_matrix.conf.
#
#  Functions: update_ngc_matrix
#  Depends:   lib/logging.sh, curl, python3 (for HTML parsing)
#
#  The scraper is best-effort: if the page structure changes or the
#  network is unreachable, the existing conf file is left untouched.
# ===========================================================================

[[ -n "${_LIB_NGC_UPDATER_LOADED:-}" ]] && return 0
_LIB_NGC_UPDATER_LOADED=1

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_LIB_DIR}/logging.sh"

_RELEASE_NOTES_INDEX_URL="https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html"
_RELEASE_NOTES_BASE_URL="https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes"

# ---------------------------------------------------------------------------
#  _scrape_compat_matrix
#  Downloads the latest Triton release-notes page and extracts the generic
#  "NVIDIA Triton Inference Server Container Versions" table into this
#  project's matrix format.
#  Outputs lines to stdout.
#  Returns 1 on any failure (network, parse, empty result).
# ---------------------------------------------------------------------------
_scrape_compat_matrix() {
    local tmp_index tmp_release release_href release_url
    tmp_index=$(mktemp /tmp/ngc_release_index.XXXXXX) || return 1
    tmp_release=$(mktemp /tmp/ngc_release_notes.XXXXXX) || {
        rm -f "$tmp_index"
        return 1
    }

    if ! curl -fsSL --connect-timeout 15 --max-time 30 \
        -o "$tmp_index" "$_RELEASE_NOTES_INDEX_URL" 2>/dev/null; then
        log_warn "Failed to fetch release-notes index"
        rm -f "$tmp_index" "$tmp_release"
        return 1
    fi

    release_href=$(
        python3 - "$tmp_index" <<'PYEOF' || true
import re
import sys
from pathlib import Path

html = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
links = re.findall(r'href="([^"]*rel[_-](\d{2})-(\d{2})\.html)[^"]*"', html)
if not links:
    sys.exit(1)

def key(item):
    _href, yy, mm = item
    return int(yy), int(mm)

print(sorted(links, key=key, reverse=True)[0][0])
PYEOF
    )
    if [ -z "$release_href" ]; then
        log_warn "Could not locate latest Triton release notes page"
        rm -f "$tmp_index" "$tmp_release"
        return 1
    fi

    if [[ "$release_href" =~ ^https?:// ]]; then
        release_url="$release_href"
    else
        release_href="${release_href%%#*}"
        release_url="${_RELEASE_NOTES_BASE_URL}/${release_href#./}"
    fi

    if ! curl -fsSL --connect-timeout 15 --max-time 30 \
        -o "$tmp_release" "$release_url" 2>/dev/null; then
        log_warn "Failed to fetch release notes: $release_url"
        rm -f "$tmp_index" "$tmp_release"
        return 1
    fi

    if [ ! -s "$tmp_release" ]; then
        log_warn "Empty response from release notes"
        rm -f "$tmp_index" "$tmp_release"
        return 1
    fi

    python3 - "$tmp_release" <<'PYEOF'
import sys, re
from html.parser import HTMLParser
from pathlib import Path

html = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")

CUDA_MIN_DRIVER = [
    # CUDA major.minor, optional update predicate, Linux x86_64 minimum driver
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

def normalize_cuda(value):
    m = re.search(r'(\d+\.\d+)(?:\.(\d+))?', value)
    if not m:
        return "-", None
    major_minor = m.group(1)
    patch = int(m.group(2) or 0)
    return major_minor, patch

def min_driver_for_cuda(cuda, patch):
    for major_minor, update, driver in CUDA_MIN_DRIVER:
        if cuda != major_minor:
            continue
        if update == "update3" and patch >= 3:
            return driver
        if update == "update2" and patch >= 2:
            return driver
        if update == "update1" and patch >= 1:
            return driver
        if update is None:
            return driver
    return "-"

def normalize_version(value):
    m = re.search(r'(\d+(?:\.\d+)+(?:\.post\d+)?(?:\.dev\d+)?)', value)
    return m.group(1) if m else "-"

def infer_python_version(tag):
    try:
        yy, mm = [int(x) for x in tag.split(".", 1)]
    except ValueError:
        return "3.12"
    if (yy, mm) >= (24, 11):
        return "3.12"
    return "3.10"

class TritonReleaseNotesParser(HTMLParser):
    """Extract rows from the generic Triton container versions table."""

    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.current_row = []
        self.cell_text = ""
        self.tables = []
        self.rows = []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.in_table = True
            self.rows = []
        if tag == "tr" and self.in_table:
            self.in_row = True
            self.current_row = []
        if tag in ("td", "th") and self.in_row:
            self.in_cell = True
            self.cell_text = ""

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.in_cell:
            self.in_cell = False
            self.current_row.append(self.cell_text.strip())
        if tag == "tr" and self.in_row:
            self.in_row = False
            if self.current_row:
                self.rows.append(self.current_row)
        if tag == "table" and self.in_table:
            self.tables.append(self.rows)
            self.in_table = False

    def handle_data(self, data):
        if self.in_cell:
            self.cell_text += data

parser = TritonReleaseNotesParser()
parser.feed(html)

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
    sys.exit(1)

for row in table[1:]:
    if len(row) < 2:
        continue

    m = re.match(r'(\d{2}\.\d{2})$', row[0].strip())
    if not m:
        continue
    ngc_tag = m.group(1)

    cuda_cell = next((cell for cell in row if "cuda" in cell.lower()), "")
    cuda, cuda_patch = normalize_cuda(cuda_cell)
    tensorrt = normalize_version(next((cell for cell in row if "tensorrt" in cell.lower()), ""))
    if cuda == "-" or tensorrt == "-":
        continue

    min_driver = min_driver_for_cuda(cuda, cuda_patch)
    if min_driver == "-":
        continue
    py_ver = infer_python_version(ngc_tag)

    print(f"{ngc_tag:6s}  {min_driver:7s}  {tensorrt:12s}  {cuda:5s}  {py_ver:5s}  -")
PYEOF
    local status=$?
    rm -f "$tmp_index" "$tmp_release"
    return $status
}

# ---------------------------------------------------------------------------
#  _merge_matrix <scraped_file> <existing_conf>
#  Merges scraped entries into the existing conf file.
#  - New tags are added
#  - Existing tags are updated if scraped data has more info (e.g. size)
#  - Tags only in conf (not on website) are preserved
#  - Result is sorted newest-first
#  Outputs merged content to stdout.
# ---------------------------------------------------------------------------
_merge_matrix() {
    local scraped_file="$1"
    local existing_conf="$2"

    python3 - "$scraped_file" "$existing_conf" <<'PYEOF'
import sys, re
from collections import OrderedDict

def parse_conf(path):
    """Parse conf file into {tag: (line, fields)} preserving comments."""
    entries = OrderedDict()
    comments = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                comments.append(line.rstrip())
                continue
            parts = stripped.split()
            if len(parts) >= 4:
                tag = parts[0]
                entries[tag] = parts
    return comments, entries

def tag_sort_key(tag):
    """Sort key for NGC tags: 26.02 > 25.11 > 25.08 etc."""
    parts = tag.split(".")
    return (int(parts[0]), int(parts[1]))

scraped_file, existing_conf = sys.argv[1], sys.argv[2]

# Parse scraped data
scraped = OrderedDict()
with open(scraped_file) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 4:
            scraped[parts[0]] = parts

# Parse existing conf
comments, existing = parse_conf(existing_conf)

# Merge: release-notes data wins for the version fields. Existing size is
# preserved when the release-notes table does not publish image size.
merged = OrderedDict()
all_tags = set(list(scraped.keys()) + list(existing.keys()))

for tag in all_tags:
    s = scraped.get(tag)
    e = existing.get(tag)

    if s and e:
        # Prefer freshly scraped release-note values for tag/min_driver/
        # TensorRT/CUDA/Python. Preserve existing size if scraped size is "-".
        result = list(s)
        s_padded = list(s)
        while len(s_padded) < 6:
            s_padded.append("-")
        e_padded = list(e)
        while len(e_padded) < 6:
            e_padded.append("-")

        result = s_padded[:6]
        if result[5] == "-" and e_padded[5] != "-":
            result[5] = e_padded[5]
        merged[tag] = result
    elif s:
        while len(s) < 6:
            s.append("-")
        merged[tag] = s
    else:
        while len(e) < 6:
            e.append("-")
        merged[tag] = e

# Sort newest-first
sorted_tags = sorted(merged.keys(), key=tag_sort_key, reverse=True)

# Output: comments then sorted entries
for c in comments:
    # Update the "Last updated" line
    if "Last updated" in c:
        from datetime import date
        c = f"#  Last updated: {date.today().isoformat()} (update-matrix)"
    print(c)

for tag in sorted_tags:
    fields = merged[tag]
    # Format with consistent spacing
    print(f"{fields[0]:6s}  {fields[1]:7s}  {fields[2]:12s}  {fields[3]:5s}  {fields[4]:5s}  {fields[5]}")
PYEOF
}

# ---------------------------------------------------------------------------
#  update_ngc_matrix [conf_path]
#  Main entry point.  Fetches the latest data from the NVIDIA compatibility
#  page and merges it into the local matrix conf file.
#  The existing file is never overwritten on failure.
# ---------------------------------------------------------------------------
update_ngc_matrix() {
    local conf_path="${1:-}"

    if [ -z "$conf_path" ]; then
        local script_dir
        script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
        conf_path="${script_dir}/ngc_matrix.conf"
    fi

    log_step "Updating NGC compatibility matrix"
    log_info "Source index: $_RELEASE_NOTES_INDEX_URL"
    log_info "Target: $conf_path"

    if [ ! -f "$conf_path" ]; then
        log_error "Matrix conf not found: $conf_path"
        log_error "Expected at: scripts/bash/ngc_matrix.conf"
        return 1
    fi

    local tmp_scraped
    tmp_scraped=$(mktemp /tmp/ngc_scraped.XXXXXX) || return 1

    if _scrape_compat_matrix > "$tmp_scraped" 2>/dev/null; then
        local count
        count=$(wc -l < "$tmp_scraped" | tr -d ' ')
        if [ "$count" -eq 0 ]; then
            log_warn "Scraper returned no entries — page structure may have changed"
            log_info "Existing matrix preserved unchanged"
            rm -f "$tmp_scraped"
            return 1
        fi
        log_info "Scraped $count entries from Triton release notes"
    else
        log_warn "Failed to scrape Triton release notes (network or parse error)"
        log_info "Existing matrix preserved unchanged"
        rm -f "$tmp_scraped"
        return 1
    fi

    local tmp_merged
    tmp_merged=$(mktemp /tmp/ngc_merged.XXXXXX) || { rm -f "$tmp_scraped"; return 1; }

    if _merge_matrix "$tmp_scraped" "$conf_path" > "$tmp_merged" 2>/dev/null; then
        local new_count old_count
        new_count=$(grep -v '^\s*#' "$tmp_merged" | grep -v '^\s*$' | wc -l | tr -d ' ')
        old_count=$(grep -v '^\s*#' "$conf_path" | grep -v '^\s*$' | wc -l | tr -d ' ')

        cp "$conf_path" "${conf_path}.bak"
        mv "$tmp_merged" "$conf_path"

        if [ "$new_count" -gt "$old_count" ]; then
            local added=$((new_count - old_count))
            log_info "Matrix updated: $old_count → $new_count entries (+$added new)"
        elif [ "$new_count" -eq "$old_count" ]; then
            log_info "Matrix up to date ($new_count entries, sizes/versions may have been refreshed)"
        else
            log_warn "Matrix shrank: $old_count → $new_count (check ${conf_path}.bak)"
        fi

        log_info "Backup saved: ${conf_path}.bak"
    else
        log_warn "Merge failed — existing matrix preserved"
        rm -f "$tmp_merged"
    fi

    rm -f "$tmp_scraped"
}
