"""Render the ControlMT showcase banner from an SVG template + JSON values.

Usage:
    # Pre-release (default values file):
    python scripts/render_showcase.py

    # Custom values file:
    python scripts/render_showcase.py --values assets/showcase_values_v2.2.json

    # Public-release (flip the ribbon and output filename):
    python scripts/render_showcase.py --release

    # Custom output:
    python scripts/render_showcase.py --output /tmp/banner.png --size 1080x1920

Notes:
    - Requires `rsvg-convert` (apt: librsvg2-bin) and Noto Sans Kannada fonts.
    - Values are simple {{KEY}} substitutions in the template — no logic, easy
      to diff and version. Edit assets/showcase_values_*.json between renders.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS = PROJECT_ROOT / "assets"
TEMPLATE = ASSETS / "showcase_template.svg"
DEFAULT_VALUES = ASSETS / "showcase_values_v2.2.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "showcase.png"


def render(template_path: Path, values: dict, out_svg: Path) -> None:
    """Substitute {{KEY}} placeholders in the SVG template, write to out_svg."""
    text = template_path.read_text(encoding="utf-8")

    def repl(match: re.Match) -> str:
        key = match.group(1)
        if key not in values:
            print(f"  ! missing key in values: {{{{{key}}}}}", file=sys.stderr)
            return match.group(0)
        # Escape for XML: &, <, >
        v = str(values[key])
        return v.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    rendered = re.sub(r"\{\{([A-Z0-9_]+)\}\}", repl, text)
    out_svg.write_text(rendered, encoding="utf-8")


def svg_to_png(svg_path: Path, png_path: Path, width: int, height: int) -> None:
    cmd = [
        "rsvg-convert",
        "--width", str(width),
        "--height", str(height),
        "--format", "png",
        "--output", str(png_path),
        str(svg_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", default=str(DEFAULT_VALUES),
                    help="JSON file with template values")
    ap.add_argument("--template", default=str(TEMPLATE))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT),
                    help="PNG output path (default: project root /showcase.png)")
    ap.add_argument("--size", default="1080x1920", help="WxH (default: 1080x1920)")
    ap.add_argument("--release", action="store_true",
                    help="Flip the ribbon to 'OFFICIAL RELEASE' (else 'PRE-RELEASE')")
    ap.add_argument("--keep-svg", action="store_true",
                    help="Keep the intermediate filled SVG next to the PNG")
    args = ap.parse_args()

    values_path = Path(args.values)
    if not values_path.exists():
        print(f"✗ values file not found: {values_path}")
        return 1
    values = json.loads(values_path.read_text(encoding="utf-8"))

    if args.release:
        values["RIBBON_TEXT"] = "OFFICIAL RELEASE"

    w, h = map(int, args.size.lower().split("x"))
    out_png = Path(args.output)
    out_svg = out_png.with_suffix(".svg")

    print(f"  template:  {args.template}")
    print(f"  values:    {values_path}")
    print(f"  ribbon:    {values.get('RIBBON_TEXT')}")
    print(f"  output:    {out_png}  ({w}×{h})")

    render(Path(args.template), values, out_svg)
    svg_to_png(out_svg, out_png, w, h)

    if not args.keep_svg:
        out_svg.unlink(missing_ok=True)

    print(f"  ✓ rendered {out_png} ({out_png.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
