#!/usr/bin/env python3
"""Render the published artifact pages from their data files.

Inlines report_data.json into templates/report_template.html (placeholder
__DATA__) producing pilot_proxy_trawl.html, and policy_data.json into
templates/policy_template.html (placeholder __POLICY__) producing
dtv_masking_policy.html. The data files come from make_report_data.py and
make_policy_data.py; the rendered pages are what the published claude.ai
artifacts serve.

    python3 analysis/render_artifacts.py --data-dir DIR [--out DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

PAGES = [  # (data file, template, placeholder, rendered page)
    ("report_data.json", "report_template.html", "__DATA__",
     "pilot_proxy_trawl.html"),
    ("policy_data.json", "policy_template.html", "__POLICY__",
     "dtv_masking_policy.html"),
]


def render(data_path: Path, template_path: Path, placeholder: str,
           out_path: Path) -> int:
    template = template_path.read_text(encoding="utf-8")
    data = data_path.read_text(encoding="utf-8")
    # keep the inline JSON safe inside a <script> block
    html = template.replace(placeholder, data.replace("</", "<\\/"))
    out_path.write_text(html, encoding="utf-8")
    return len(html)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("out"),
                    help="directory holding report_data.json / policy_data.json")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: --data-dir)")
    ap.add_argument("--templates", type=Path,
                    default=Path(__file__).resolve().parent / "templates")
    ap.add_argument("--only", choices=["report", "policy"], default=None)
    args = ap.parse_args(argv)
    out_dir = args.data_dir if args.out is None else args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    for data_name, template_name, placeholder, page_name in PAGES:
        if args.only == "report" and not page_name.startswith("pilot_proxy"):
            continue
        if args.only == "policy" and not page_name.startswith("dtv_"):
            continue
        data_path = args.data_dir / data_name
        if not data_path.is_file():
            raise SystemExit(f"{data_path} not found; run "
                             "make_report_data.py / make_policy_data.py first")
        n = render(data_path, args.templates / template_name, placeholder,
                   out_dir / page_name)
        print("wrote", out_dir / page_name, n, "bytes")


if __name__ == "__main__":
    main()
