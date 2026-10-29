"""Check local Markdown links and GitHub-style heading anchors without network IO.

Defaults to docs/, README.md and finetune/OHT/README.md. Optional file/directory
arguments are relative to the repository root. Code fences and inline code are
excluded; inline links, reference definitions and HTML href/src are checked.
"""
import argparse
import html
import re
import sys
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = ("docs", "README.md", "finetune/OHT/README.md")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
INLINE_LINK = re.compile(
    r"!?\[[^\]\n]*\]\(\s*(<[^>]*>|[^\s)]+)(?:\s+['\"][^\n]*?['\"])?\s*\)"
)
REFERENCE = re.compile(r"^ {0,3}\[[^\]]+\]:\s*(<[^>]+>|[^\s]+)")
HTML_LINK = re.compile(r"\b(?:href|src)\s*=\s*(['\"])(.*?)\1", re.IGNORECASE)
HTML_ID = re.compile(r"\b(?:id|name)\s*=\s*(?:['\"]([^'\"]+)['\"]|([^\s>]+))", re.IGNORECASE)


def prose_lines(text):
    """Yield numbered lines outside fenced code blocks."""
    fence = None
    for number, line in enumerate(text.splitlines(), 1):
        match = FENCE.match(line)
        if fence is None:
            if match:
                fence = match[1]
            else:
                yield number, line
        elif (match and match[1][0] == fence[0] and len(match[1]) >= len(fence)
              and not match[2].strip()):
            fence = None


def heading_anchors(text):
    anchors, used = set(), set()
    for _, line in prose_lines(text):
        for tag in re.findall(r"<[^>]+>", line):
            for match in HTML_ID.finditer(tag):
                anchors.add(html.unescape(match[1] or match[2]))
        heading = re.match(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if not heading:
            continue
        title = re.sub(r"!?\[([^\]]+)\]\([^)]+\)", r"\1", heading[1])
        title = html.unescape(re.sub(r"<[^>]+>", "", title)).lower()
        base = "".join(c for c in title if c in "-_ " or unicodedata.category(c)[0] in "LNM")
        base = base.replace(" ", "-")
        slug, count = base, 0
        while slug in used:
            count += 1
            slug = f"{base}-{count}"
        used.add(slug)
        anchors.add(slug)
    return anchors


def link_targets(line):
    line = re.sub(r"(`+).*?\1", "", line)
    for match in INLINE_LINK.finditer(line):
        yield match[1].strip("<>")
    reference = REFERENCE.match(line)
    if reference:
        yield reference[1].strip("<>")
    for tag in re.findall(r"<[^>]+>", line):
        for match in HTML_LINK.finditer(tag):
            yield match[2]


def check_files(files):
    issues, cache, count = [], {}, 0
    for file in files:
        try:
            content = file.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as error:
            issues.append(f"{file}: cannot read UTF-8 Markdown: {error}")
            continue
        for line_number, line in prose_lines(content):
            for raw in link_targets(line):
                raw = html.unescape(raw)
                try:
                    parsed = urlsplit(raw)
                    if parsed.scheme or parsed.netloc:
                        continue
                    count += 1
                    target = (file.parent / unquote(parsed.path)).resolve() if parsed.path else file
                    if not target.exists():
                        raise ValueError("missing file")
                    if parsed.fragment and target.suffix.lower() == ".md":
                        if target not in cache:
                            cache[target] = heading_anchors(target.read_text(encoding="utf-8-sig"))
                        if unquote(parsed.fragment) not in cache[target]:
                            raise ValueError("missing heading or HTML anchor")
                except (OSError, UnicodeError, ValueError) as error:
                    issues.append(f"{file}:{line_number}: {error}: {raw}")
    return count, issues


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Markdown files/directories, relative to repository root")
    args = parser.parse_args(argv)
    files = set()
    for supplied in args.paths or DEFAULT_PATHS:
        path = (ROOT / supplied).resolve()
        if path.is_dir():
            files.update(path.rglob("*.md"))
        elif path.is_file() and path.suffix.lower() == ".md":
            files.add(path)
        else:
            parser.error(f"not a Markdown file or directory: {supplied}")
    count, issues = check_files(sorted(files))
    for issue in issues:
        print(issue, file=sys.stderr)
    print(f"Checked {len(files)} Markdown files, {count} local links; {len(issues)} issue(s).")
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
