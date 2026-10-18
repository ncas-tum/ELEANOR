"""Build the per-variant package index (PEP 503) served from GitHub Pages.

    <out>/<variant>/index.html                 project list
    <out>/<variant>/eleanor-ncas/index.html     every eleanor-ncas wheel of that variant
    <out>/<variant>/eleanor-ncas/<wheel>        the wheels themselves
    <out>/<variant>/torch/index.html           links to the matching PyTorch builds
    <out>/<variant>/torchvision/index.html     (on download.pytorch.org)

The variant comes from each wheel's local version (0.1.0+cu126 -> cu126). The
eleanor-ncas wheels pin exact torch/torchvision builds (e.g. torch==2.14.1+cu126),
which only exist on PyTorch's index, so the torch pages link to exactly those
files. Everything else (jax, numpy, the NVIDIA libraries, ...) comes from PyPI:

    pip install eleanor-ncas --extra-index-url https://<pages>/whl/cu126
    uv add eleanor-ncas --index https://<pages>/whl/cu126

Usage:
    python scripts/build_index.py -o docs/_build/whl wheels/*.whl
"""

import argparse
import hashlib
import html
import re
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

PROJECT = "eleanor-ncas"
PYTORCH_INDEX = "https://download.pytorch.org/whl"
PINNED = ("torch", "torchvision")
ANCHOR = re.compile(r"<a\s+([^>]*?)href=\"([^\"]+)\"([^>]*)>([^<]+)</a>")


def wheel_metadata(path):
    with zipfile.ZipFile(path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
        return zf.read(name).decode()


def pinned_versions(metadata):
    """{"torch": "2.14.1+cu126", ...} from the wheel's Requires-Dist pins."""
    pins = {}
    for m in re.finditer(
        r"^Requires-Dist:\s*([A-Za-z0-9._-]+)\s*==\s*([^\s;]+)\s*$",
        metadata,
        re.MULTILINE,
    ):
        if m.group(1).lower() in PINNED:
            pins[m.group(1).lower()] = m.group(2)
    return pins


def page(title, links):
    body = "\n".join(f"    {link}<br/>" for link in links)
    return (
        '<!DOCTYPE html>\n<html>\n  <head><meta name="pypi:repository-version" '
        f'content="1.0"><title>{title}</title></head>\n  <body>\n'
        f"    <h1>{title}</h1>\n{body}\n  </body>\n</html>\n"
    )


def pytorch_links(variant, project, versions):
    """Anchors from PyTorch's index for the given exact versions, with absolute URLs."""
    url = f"{PYTORCH_INDEX}/{variant}/{project}/"
    with urllib.request.urlopen(url, timeout=60) as response:
        listing = response.read().decode()
    links = []
    for before, href, after, text in ANCHOR.findall(listing):
        version = urllib.parse.unquote(text.strip()).split("-")[1]
        if version in versions:
            href = urllib.parse.urljoin(url, html.unescape(href))
            links.append(
                f'<a {before}href="{html.escape(href)}"{after}>{text.strip()}</a>'
            )
    missing = {
        v
        for v in versions
        if not any(f"-{v}-" in urllib.parse.unquote(a) for a in links)
    }
    if missing:
        sys.exit(f"error: {project} {sorted(missing)} not found on {url}")
    return links


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wheels", nargs="+", type=Path)
    parser.add_argument("-o", "--out-dir", type=Path, required=True)
    args = parser.parse_args()

    by_variant = defaultdict(list)
    for wheel in args.wheels:
        version = wheel.name.split("-")[1]
        if "+" not in version:
            sys.exit(f"error: {wheel.name} has no variant suffix (e.g. +cu126)")
        by_variant[version.split("+", 1)[1]].append(wheel)

    for variant, wheels in sorted(by_variant.items()):
        root = args.out_dir / variant
        project_dir = root / PROJECT
        project_dir.mkdir(parents=True, exist_ok=True)

        links, pins = [], defaultdict(set)
        for wheel in sorted(wheels):
            metadata = wheel_metadata(wheel)
            for project, version in pinned_versions(metadata).items():
                pins[project].add(version)
            shutil.copy2(wheel, project_dir / wheel.name)
            sha = hashlib.sha256(wheel.read_bytes()).hexdigest()
            requires = re.search(r"^Requires-Python:\s*(.+)$", metadata, re.MULTILINE)
            attr = (
                f' data-requires-python="{html.escape(requires.group(1).strip())}"'
                if requires
                else ""
            )
            href = f"{urllib.parse.quote(wheel.name)}#sha256={sha}"
            links.append(f'<a href="{href}"{attr}>{html.escape(wheel.name)}</a>')
        (project_dir / "index.html").write_text(page(f"Links for {PROJECT}", links))

        projects = [PROJECT]
        for project, versions in sorted(pins.items()):
            (root / project).mkdir(exist_ok=True)
            torch_links = pytorch_links(variant, project, versions)
            (root / project / "index.html").write_text(
                page(f"Links for {project}", torch_links)
            )
            projects.append(project)
            print(
                f"{variant}: {project} {', '.join(sorted(versions))} ({len(torch_links)} files)"
            )

        (root / "index.html").write_text(
            page(f"ELEANOR ({variant})", [f'<a href="{p}/">{p}</a>' for p in projects])
        )
        print(f"{variant}: {len(wheels)} {PROJECT} wheel(s)")


if __name__ == "__main__":
    main()
