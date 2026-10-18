"""Turn a wheel built for one hardware variant into a self-contained variant wheel.

The Docker build produces e.g. eleanor_ncas-0.1.0+cu126-cp311-abi3-....whl, with
the variant extras (cpu, cuda12, cuda13, mps) optional. This script:

- makes the requirements of the variant's extra unconditional (e.g. cuda12:
  torch, torchvision and jax[cuda12]), so a plain `pip install eleanor-ncas`
  from the variant index pulls the right builds;
- pins torch/torchvision to the exact builds the extension was compiled
  against (e.g. torch==2.14.1+cu126), since the extension only works with them;
- drops all variant extras, which are meaningless in a variant wheel.

Usage:
    python scripts/make_variant_wheel.py dist/raw.whl --variant cu126 --extra cuda12 \
        --pin torch==2.14.1+cu126 --pin torchvision==0.29.1+cu126 -o dist/variants
"""

import argparse
import base64
import hashlib
import re
import sys
import zipfile
from pathlib import Path

VARIANT_EXTRAS = ("cpu", "cuda12", "cuda13", "mps")
EXTRA_REQ = re.compile(
    r"^Requires-Dist:\s*(.+?)\s*;\s*extra\s*==\s*[\"']([^\"']+)[\"']\s*$"
)
PROVIDES = re.compile(r"^Provides-Extra:\s*(\S+)\s*$")


def req_name(requirement):
    """Normalized project name of a requirement string ("jax[cuda12]>=0.10" -> "jax")."""
    name = re.match(r"[A-Za-z0-9._-]+", requirement).group(0)
    return re.sub(r"[-_.]+", "-", name).lower()


def rewrite_metadata(metadata, extra, pins):
    pins = {req_name(p): p for p in pins}
    lines = metadata.decode().splitlines()
    header_end = lines.index("") if "" in lines else len(lines)
    header, body = lines[:header_end], lines[header_end:]

    promoted, kept = [], []
    for line in header:
        m = EXTRA_REQ.match(line)
        if m and m.group(2) == extra:
            promoted.append(m.group(1))
        elif (m and m.group(2) in VARIANT_EXTRAS) or (
            PROVIDES.match(line) and PROVIDES.match(line).group(1) in VARIANT_EXTRAS
        ):
            continue
        else:
            if "extra ==" in line and line.startswith("Requires-Dist:") and not m:
                sys.exit(f"error: unsupported marker in {line!r}")
            kept.append(line)
    if not promoted:
        sys.exit(f"error: the wheel has no requirements for extra {extra!r}")

    requirements = []
    for req in promoted:
        requirements.append(pins.pop(req_name(req), req))
    if pins:
        sys.exit(f"error: pinned packages not in extra {extra!r}: {sorted(pins)}")

    # Insert the new requirements after the last unconditional Requires-Dist
    last = max(
        (
            i
            for i, line in enumerate(kept)
            if line.startswith("Requires-Dist:") and "extra ==" not in line
        ),
        default=max(i for i, line in enumerate(kept) if line.startswith("Version:")),
    )
    kept[last + 1 : last + 1] = [f"Requires-Dist: {r}" for r in requirements]
    return ("\n".join(kept + body) + "\n").encode(), requirements


def record_hash(data):
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"sha256={digest.decode()}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wheel", type=Path)
    parser.add_argument(
        "--variant", required=True, help="local version suffix, e.g. cu126"
    )
    parser.add_argument("--extra", required=True, choices=VARIANT_EXTRAS)
    parser.add_argument(
        "--pin", action="append", default=[], help="e.g. torch==2.14.1+cu126"
    )
    parser.add_argument("-o", "--out-dir", type=Path, required=True)
    args = parser.parse_args()

    version = args.wheel.name.split("-")[1]
    if not version.endswith(f"+{args.variant}"):
        sys.exit(f"error: wheel version {version} does not end with +{args.variant}")

    with zipfile.ZipFile(args.wheel) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        files = {i.filename: (zf.read(i), i.external_attr) for i in infos}
    dist_info = {
        n.split("/")[0] for n in files if n.split("/")[0].endswith(".dist-info")
    }
    if len(dist_info) != 1:
        sys.exit(f"error: expected one .dist-info directory, found {sorted(dist_info)}")
    dist_info = dist_info.pop()

    metadata_path = f"{dist_info}/METADATA"
    data, attr = files[metadata_path]
    metadata, requirements = rewrite_metadata(data, args.extra, args.pin)
    files[metadata_path] = (metadata, attr)

    record_path = f"{dist_info}/RECORD"
    files.pop(record_path, None)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / args.wheel.name
    record = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        # dist-info goes last, as recommended by the wheel spec
        for path in sorted(files, key=lambda p: (p.startswith(dist_info), p)):
            data, attr = files[path]
            info = zipfile.ZipInfo(path, (1980, 1, 1, 0, 0, 0))
            info.external_attr = attr  # keeps file permissions
            zf.writestr(info, data, zipfile.ZIP_DEFLATED)
            record.append(f"{path},{record_hash(data)},{len(data)}")
        record.append(f"{record_path},,")
        info = zipfile.ZipInfo(record_path, (1980, 1, 1, 0, 0, 0))
        info.external_attr = 0o644 << 16
        zf.writestr(info, "\n".join(record) + "\n", zipfile.ZIP_DEFLATED)

    print(f"Wrote {out}")
    for r in requirements:
        print(f"  requires: {r}")


if __name__ == "__main__":
    main()
