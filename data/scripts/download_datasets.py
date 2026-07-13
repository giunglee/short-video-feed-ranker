#!/usr/bin/env python3
"""Download public KuaiRec / KuaiRand datasets into data/raw/.

Sources (official):
  KuaiRec:   https://kuairec.com/  · Zenodo https://zenodo.org/records/18164998
  KuaiRand:  https://kuairand.com/

Usage:
    python data/scripts/download_datasets.py --all
    python data/scripts/download_datasets.py --kuairec
    python data/scripts/download_datasets.py --kuairand-pure --kuairand-1k
    python data/scripts/download_datasets.py --kuairec --extras   # optional CSVs

Skip KuaiRand-27K (~46 GB uncompressed) — not needed for this repo.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

KUAIREC_ZIP = "https://zenodo.org/records/18164998/files/KuaiRec.zip"
KUAIREC_EXTRAS: dict[str, str] = {
    "user_features_raw.csv": (
        "https://zenodo.org/records/18164998/files/user_features_raw.csv"
    ),
    "video_raw_categories_multi.csv": (
        "https://zenodo.org/records/18164998/files/video_raw_categories_multi.csv"
    ),
    "kuairec_caption_category.csv": (
        "https://zenodo.org/records/18164998/files/kuairec_caption_category.csv"
    ),
    "item_daily_features.csv": (
        "https://zenodo.org/records/18164998/files/item_daily_features.csv"
    ),
}

KUAIRAND_TARBALLS: dict[str, tuple[str, str]] = {
  # key -> (url, extract subdir name inside tarball)
    "pure": (
        "https://chongming.myds.me:61364/data/KuaiRand-Pure.tar.gz",
        "KuaiRand-Pure",
    ),
    "1k": (
        "https://chongming.myds.me:61364/data/KuaiRand-1K.tar.gz",
        "KuaiRand-1K",
    ),
}

KUAIRAND_SUPPLEMENT: dict[str, str] = {
    "kuairand_video_categories.csv": (
        "https://zenodo.org/records/18159199/files/kuairand_video_categories.csv"
    ),
    "kuairand_video_captions.csv": (
        "https://zenodo.org/records/18159199/files/kuairand_video_captions.csv"
    ),
}


def _download(url: str, dest: Path, *, label: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  skip (exists): {dest.name}")
        return
    print(f"  downloading {label} ...")
    print(f"    {url}")

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        done = block_num * block_size
        pct = min(100, done * 100 // total_size)
        mb = done / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\r    {pct:3d}% ({mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def _copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                _copy_tree(item, target)
            else:
                shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def download_kuairec(*, extras: bool = False) -> None:
    out_dir = RAW / "kuairec"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== KuaiRec ===")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "KuaiRec.zip"
        _download(KUAIREC_ZIP, archive, label="KuaiRec.zip")
        print("  extracting KuaiRec.zip ...")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(Path(tmp))

        # Official zip may nest files under KuaiRec/ or at archive root.
        candidates = [Path(tmp) / "KuaiRec", Path(tmp)]
        for base in candidates:
            if (base / "small_matrix.csv").exists() or (base / "big_matrix.csv").exists():
                for item in base.iterdir():
                    if item.name in {".gitkeep", "KuaiRec.zip"}:
                        continue
                    target = out_dir / item.name
                    if item.is_dir():
                        if target.exists():
                            _copy_tree(item, target)
                        else:
                            shutil.copytree(item, target)
                    else:
                        shutil.copy2(item, target)
                break

    if extras:
        print("  optional supplementary CSVs ...")
        for name, url in KUAIREC_EXTRAS.items():
            _download(url, out_dir / name, label=name)

    print(f"  done → {out_dir.relative_to(ROOT)}/")


def download_kuairand_variant(key: str, *, supplement: bool = False) -> None:
    url, folder_name = KUAIRAND_TARBALLS[key]
    out_dir = RAW / f"kuairand_{key}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== KuaiRand-{key.upper()} ===")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / f"KuaiRand-{key}.tar.gz"
        _download(url, archive, label=archive.name)
        print("  extracting ...")
        with tarfile.open(archive, "r:gz") as tf:
            # filter= added in Python 3.12; safe fallback for 3.10+
            try:
                tf.extractall(Path(tmp), filter="data")
            except TypeError:
                tf.extractall(Path(tmp))

        data_dir = Path(tmp) / folder_name / "data"
        if not data_dir.is_dir():
            # fallback: search for log_*.csv
            matches = list(Path(tmp).rglob("log_standard*.csv"))
            data_dir = matches[0].parent if matches else Path(tmp)

        for item in data_dir.iterdir():
            if item.name == ".gitkeep":
                continue
            shutil.copy2(item, out_dir / item.name)

        license_src = Path(tmp) / folder_name / "LICENSE"
        if license_src.is_file():
            shutil.copy2(license_src, out_dir / "LICENSE")

    if supplement:
        print("  optional supplementary CSVs (large) ...")
        for name, file_url in KUAIRAND_SUPPLEMENT.items():
            _download(file_url, out_dir / name, label=name)

    print(f"  done → {out_dir.relative_to(ROOT)}/")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download KuaiRec / KuaiRand datasets into data/raw/."
    )
    parser.add_argument("--all", action="store_true", help="KuaiRec + Pure + 1K")
    parser.add_argument("--kuairec", action="store_true", help="KuaiRec core zip")
    parser.add_argument(
        "--extras",
        action="store_true",
        help="Also fetch optional KuaiRec supplementary CSVs from Zenodo",
    )
    parser.add_argument("--kuairand-pure", action="store_true", help="KuaiRand-Pure")
    parser.add_argument("--kuairand-1k", action="store_true", help="KuaiRand-1K")
    parser.add_argument(
        "--supplement",
        action="store_true",
        help="Also fetch optional KuaiRand category/caption CSVs (multi-GB)",
    )
    args = parser.parse_args(argv)

    if args.all:
        args.kuairec = True
        args.kuairand_pure = True
        args.kuairand_1k = True

    if not (args.kuairec or args.kuairand_pure or args.kuairand_1k):
        parser.print_help()
        print(
            "\nNo dataset selected. Example: python data/scripts/download_datasets.py --all",
            file=sys.stderr,
        )
        return 1

    if args.kuairec:
        download_kuairec(extras=args.extras)
    if args.kuairand_pure:
        download_kuairand_variant("pure", supplement=args.supplement)
    if args.kuairand_1k:
        download_kuairand_variant("1k", supplement=args.supplement)

    print("\nAll requested downloads finished.")
    print("Dataset docs: https://kuairec.com · https://kuairand.com")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
