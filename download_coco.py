"""
Download the COCO 2014 val image set and the full annotations zip
required by chair.py to rebuild its CHAIR evaluator.

Outputs:
    <output-dir>/coco/val2014/*.jpg
    <output-dir>/coco_annotations/{captions,instances}_{train,val}2014.json

Usage:
    python download_coco.py [--output-dir .] [--skip-images] [--skip-annotations]
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


COCO_FILES = {
    "val2014": {
        "url": "http://images.cocodataset.org/zips/val2014.zip",
        "approx_size_mb": 6350,
    },
    "annotations_trainval2014": {
        "url": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
        "approx_size_mb": 252,
    },
}


def download_with_resume(url: str, dst: Path, chunk_size: int = 1024 * 1024) -> None:
    """Stream-download `url` to `dst` with HTTP Range resume support."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    resume_pos = dst.stat().st_size if dst.exists() else 0
    headers = {"Range": f"bytes={resume_pos}-"} if resume_pos else {}

    with requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + resume_pos
        mode = "ab" if resume_pos else "wb"
        with open(dst, mode) as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=dst.name,
            initial=resume_pos,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def unzip(zip_path: Path, out_dir: Path) -> None:
    """Extract `zip_path` into `out_dir`, stripping the top-level dir if any."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            # COCO zips wrap files in a single top-level dir like "val2014/".
            parts = Path(member.filename).parts
            if len(parts) > 1:
                rel = Path(*parts[1:])
            else:
                rel = Path(parts[0])
            if str(rel) in ("", "."):
                continue
            target = out_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--skip-annotations", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Where to keep the raw .zip files. Default: <output-dir>/.cache")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.output_dir / ".cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    if not args.skip_images:
        jobs.append(("val2014", args.output_dir / "coco" / "val2014",
                     cache_dir / "val2014.zip"))
    if not args.skip_annotations:
        jobs.append(("annotations_trainval2014", args.output_dir / "coco_annotations",
                     cache_dir / "annotations_trainval2014.zip"))

    for key, extract_dir, zip_path in jobs:
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(f"[skip] {key}: already extracted at {extract_dir}")
            continue

        meta = COCO_FILES[key]
        # Always call download_with_resume: it picks up the existing partial
        # file via HTTP Range when one exists, so re-running after a crash
        # resumes rather than restarting from 0.
        if zip_path.exists() and zip_path.stat().st_size >= 1024 * 1024:
            print(f"[resume]  {key}: existing partial at {zip_path} ({zip_path.stat().st_size // (1024*1024)} MB)")
        else:
            print(f"[download] {key} from {meta['url']} (≈{meta['approx_size_mb']} MB)")
        download_with_resume(meta["url"], zip_path)

        print(f"[extract]  {key} -> {extract_dir}")
        unzip(zip_path, extract_dir)
        print(f"[done]     {key}")

    # Sanity checks
    val2014_dir = args.output_dir / "coco" / "val2014"
    if val2014_dir.exists():
        n = sum(1 for _ in val2014_dir.glob("COCO_val2014_*.jpg"))
        print(f"[check] val2014 contains {n} jpg images (expected 40504)")
        if n < 40000:
            print(f"[warn] val2014 has {n} images, far below the expected 40504. "
                  "Re-run without --skip-images.")

    ann_dir = args.output_dir / "coco_annotations"
    required = [
        "captions_train2014.json",
        "captions_val2014.json",
        "instances_train2014.json",
        "instances_val2014.json",
    ]
    if ann_dir.exists():
        missing = [f for f in required if not (ann_dir / f).exists()]
        if missing:
            print(f"[error] coco_annotations is missing: {missing}")
            return 1
        print(f"[check] coco_annotations has all 4 required files")

    return 0


if __name__ == "__main__":
    sys.exit(main())
