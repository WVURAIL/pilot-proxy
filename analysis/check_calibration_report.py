#!/usr/bin/env python3
"""Static checks on the rendered report: tags, placeholders, theming, assets."""
from __future__ import annotations

import os
import re
import sys
from html.parser import HTMLParser

import _calibration_paths as P

PATH = os.path.join(str(P.OUT), "calibration_report.html")
VOID = {"img", "br", "hr", "meta", "link", "input", "source", "col"}


class Balance(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append((tag, self.getpos()))

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack:
            self.errors.append("stray </%s> at %s" % (tag, self.getpos()))
            return
        top, pos = self.stack.pop()
        if top != tag:
            self.errors.append("closed </%s> but <%s> was open (opened %s, "
                               "closed %s)" % (tag, top, pos, self.getpos()))


def main():
    src = open(PATH, encoding="utf-8").read()
    fail = []

    left = set(re.findall(r"\$\{(\w+)\}", src))
    if left:
        fail.append("unfilled placeholders: %s" % sorted(left))

    b = Balance()
    b.feed(src)
    if b.errors:
        fail.append("tag balance: %s" % b.errors[:6])
    if b.stack:
        fail.append("unclosed: %s" % [t for t, _ in b.stack][:8])

    imgs = re.findall(r'<img src="data:image/png;base64,([A-Za-z0-9+/=]{0,80})',
                      src)
    if len(imgs) < 12:
        fail.append("only %d embedded images" % len(imgs))
    if any(len(x) < 40 for x in imgs):
        fail.append("an embedded image looks truncated")

    for probe in (":root{", "@media (prefers-color-scheme: dark)",
                  ':root:not([data-theme="light"])', ':root[data-theme="dark"]'):
        if probe not in src:
            fail.append("missing theme block: %s" % probe)
    if not re.search(r"body\{[^}]*background:var\(--paper\)", src):
        fail.append("body does not set an explicit token background")

    css = src[src.index("<style>"):src.index("</style>")]
    dark_block = re.search(
        r':root:not\(\[data-theme="light"\]\)\{(.*?)\}', css, re.S)
    root_block = re.search(r":root\{(.*?)\}", css, re.S)
    if dark_block and root_block:
        dark_vars = set(re.findall(r"(--[\w-]+):", dark_block.group(1)))
        root_vars = set(re.findall(r"(--[\w-]+):", root_block.group(1)))
        orphan = dark_vars - root_vars
        if orphan:
            fail.append("tokens defined only in the dark block: %s"
                        % sorted(orphan))

    m = re.search(r"<title>(.*?)</title>", src)
    if not m:
        fail.append("no <title>")
    elif len(m.group(1)) > 40:
        fail.append("title too long: %r" % m.group(1))

    n_tables = src.count("<table")
    n_scroll = src.count('class="scroll"')
    if n_scroll < n_tables:
        fail.append("%d tables but %d scroll containers" % (n_tables, n_scroll))

    hosts = set(re.findall(r'https?://([\w.-]+)', src))
    allowed = {"fonts.googleapis.com", "fonts.gstatic.com"}
    if hosts - allowed:
        fail.append("external hosts beyond Google Fonts: %s"
                    % sorted(hosts - allowed))

    secs = re.findall(r'sec-n">(\d+)<', src)
    if secs != [("%02d" % i) for i in range(1, len(secs) + 1)]:
        fail.append("section numbers are not 01..N in order: %s" % secs)

    print("size          : %.2f MB" % (os.path.getsize(PATH) / 1048576))
    print("embedded figs : %d" % len(imgs))
    print("sections      : %s" % ", ".join(secs))
    print("tables        : %d (all in scroll containers: %s)"
          % (n_tables, n_scroll >= n_tables))
    print("title         : %r" % (m.group(1) if m else None))
    print("external hosts: %s" % sorted(hosts))
    print()
    if fail:
        print("FAIL")
        for f in fail:
            print("  -", f)
        return 1
    print("PASS -- all structural checks clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
