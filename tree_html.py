#!/usr/bin/env python3
"""
dir_tree_html.py  – Generate a pretty, chronologically-sorted HTML “tree view”
                    (newest‐modified items first) of any directory,
                    with optional file/folder filters.

Usage
-----
    python dir_tree_html.py <base_dir> [-o output.html]
        [--filter-file FILE [FILE ...]]
        [--filter-folder FOLDER [FOLDER ...]]

• For **files**: shows creation date, modified date, and size.
• For **folders**: shows creation date, modified date, and *cumulative* size
  of everything inside the folder, plus a collapsible sub-tree.
• **filter-file**: list of filenames to skip (default: tmp.py, tmp.c)
• **filter-folder**: list of folder names to skip (default: __pycache__)

The output is a self-contained HTML file with embedded CSS (dark-mode aware)
and zero external dependencies.
"""

from __future__ import annotations
import argparse
import html
import os
import sys
from datetime import datetime
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────
DT_FMT = "%Y-%m-%d %H:%M:%S"

def ts_to_str(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime(DT_FMT)

def human_size(num_bytes: int) -> str:
    if num_bytes == 1:
        return "1 byte"
    for unit in ("bytes", "KiB", "MiB", "GiB", "TiB"):
        if num_bytes < 1024.0 or unit == "TiB":
            return f"{num_bytes:.1f} {unit}" if unit != "bytes" else f"{num_bytes} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PiB"

# ──────────────────────────────────────────────────────────────────────────────
# Tree model
# ──────────────────────────────────────────────────────────────────────────────
class Node:
    __slots__ = ("path","name","is_dir","size","ctime","mtime","children",
                 "filter_file","filter_folder")

    def __init__(
        self,
        path: Path,
        is_dir: bool,
        size: int,
        ctime: float,
        mtime: float,
        children: list["Node"] | None = None,
        filter_file: set[str] = set(),
        filter_folder: set[str] = set(),
    ):
        self.path = path
        self.name = path.name or str(path)
        self.is_dir = is_dir
        self.size = size
        self.ctime = ctime
        self.mtime = mtime
        self.children = children or []
        self.filter_file = filter_file
        self.filter_folder = filter_folder

    @classmethod
    def build(
        cls,
        path: Path,
        filter_file: set[str],
        filter_folder: set[str],
    ) -> "Node":
        try:
            st = path.stat()
        except OSError as e:
            print(f"⚠️  Skipping {path} – {e}", file=sys.stderr)
            return cls(path, False, 0, 0, 0, None, filter_file, filter_folder)

        name = path.name
        # apply filters
        if path.is_dir() and name in filter_folder:
            print(f"🔕  Skipping folder {path}", file=sys.stderr)
            return cls(path, True, 0, st.st_ctime, st.st_mtime, [], filter_file, filter_folder)
        if not path.is_dir() and name in filter_file:
            print(f"🔕  Skipping file {path}", file=sys.stderr)
            return cls(path, False, 0, st.st_ctime, st.st_mtime, None, filter_file, filter_folder)

        if path.is_dir():
            try:
                raw_entries = list(path.iterdir())
            except OSError as e:
                print(f"⚠️  Cannot list {path} – {e}", file=sys.stderr)
                raw_entries = []

            stat_map: dict[Path, os.stat_result] = {}
            for child in raw_entries:
                try:
                    stat_map[child] = child.stat()
                except OSError as e:
                    print(f"⚠️  Skipping {child} – {e}", file=sys.stderr)

            entries = [p for p in raw_entries if p in stat_map]
            entries.sort(key=lambda p: stat_map[p].st_mtime, reverse=True)

            children: list[Node] = []
            total = 0
            for child in entries:
                node = cls.build(child, filter_file, filter_folder)
                # only count size if not filtered
                total += node.size
                children.append(node)

            return cls(path, True, total, st.st_ctime, st.st_mtime, children, filter_file, filter_folder)
        else:
            return cls(path, False, st.st_size, st.st_ctime, st.st_mtime, None, filter_file, filter_folder)

    def to_html(self, indent: int = 0) -> str:
        esc = html.escape(self.name, quote=False)
        created = ts_to_str(self.ctime)
        modified = ts_to_str(self.mtime)
        size_h = human_size(self.size)
        pad = "  " * indent

        if self.is_dir:
            header = (
                f"{pad}<details open>\n"
                f"{pad}  <summary>📁 <strong>{esc}/</strong>"
                f" ({size_h}, modified {modified})</summary>\n"
                f"{pad}  <ul>\n"
            )
            body = "".join(child.to_html(indent + 2) for child in self.children)
            footer = f"{pad}  </ul>\n{pad}</details>\n"
            return header + body + footer
        else:
            return (
                f"{pad}<li>📄 {esc} "
                f"<small>({size_h}, modified {modified})</small></li>\n"
            )

# ──────────────────────────────────────────────────────────────────────────────
# HTML wrapper (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
DOC_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Directory tree for {root}</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root {{
  color-scheme: light dark;
  --font-stack: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica,
                Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  padding: 1.5rem;
  font-family: var(--font-stack);
  line-height: 1.5;
  font-size: 0.95rem;
}}
ul {{ list-style-type: none; padding-left: 1em; margin: 0; }}
li {{ margin: 0.2em 0; }}
details > summary {{
  cursor: pointer;
  margin: 0.4em 0;
}}
summary::-webkit-details-marker {{ display: none; }}
details summary::before {{
  content: "▸ ";
  display: inline-block;
  transition: transform 0.1s ease-in-out;
}}
details[open] summary::before {{
  transform: rotate(90deg);
}}
strong {{ font-weight: 600; }}
small {{ opacity: 0.75; }}
@media (prefers-color-scheme: dark) {{
  body {{ background: #111; color: #ddd; }}
  a   {{ color: #8ab4f8; }}
}}
</style>
</head>
<body>
<h1>Directory tree for <code>{root}</code></h1>
<blockquote>
Generated {now} (local time). Entries sorted by <em>most recently modified</em>.
</blockquote>
<main>
{tree_html}
</main>
</body>
</html>
"""

def build_html_document(root_path: Path, tree_html: str) -> str:
    return DOC_TEMPLATE.format(
        root=html.escape(str(root_path)),
        now=datetime.now().strftime(DT_FMT),
        tree_html=tree_html,
    )

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a prettified HTML tree view of a directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("base_dir", type=Path, help="Directory to scan")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("dir_tree.html"),
        help="Destination HTML file",
    )
    parser.add_argument(
        "--filter-file", nargs="*", default=[],
        help="List of filenames to skip"
    )
    parser.add_argument(
        "--filter-folder", nargs="*", default=["__pycache__", "wandb", '.vscode-server', '.git', '.vscode'],
        help="List of folder names to skip"
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.expanduser().resolve()
    if not base_dir.exists():
        sys.exit(f"❌  Path does not exist: {base_dir}")

    # convert lists to sets for faster lookup
    ff = set(args.filter_file)
    fd = set(args.filter_folder)

    print(f"📂 Building tree for {base_dir} …")
    root_node = Node.build(base_dir, ff, fd)

    print("🖋️  Rendering HTML …")
    html_tree = root_node.to_html(indent=0)
    html_doc = build_html_document(base_dir, html_tree)

    args.output.write_text(html_doc, encoding="utf-8")
    print(f"✅  Done!  Open {args.output} in a browser.")

if __name__ == "__main__":
    main()