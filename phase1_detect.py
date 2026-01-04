#!/usr/bin/env python3
"""
MeteoFinder Phase 1 - Local OpenCV pre-filtering only (NO API USAGE)

This script performs local detection using OpenCV and saves candidates to the Candidates/ folder.
Run Phase 2 separately to verify candidates with Claude Vision API.

Usage:
    python phase1_detect.py <folder_path> [--sensitivity N]

Examples:
    python phase1_detect.py D:\\Photos\\NightSky
    python phase1_detect.py ./images --sensitivity 2
    python phase1_detect.py ./images -s 4
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from config import SUPPORTED_EXTENSIONS, DEFAULT_SENSITIVITY
from detector.prefilter import detect_streak, get_sensitivity_description


CANDIDATES_FOLDER = "Candidates"  # Output folder for Phase 1


def find_images(folder: Path, exclude_folders: List[str] = None) -> List[Path]:
    """Find all supported image files in a folder, excluding specified subfolders."""
    if exclude_folders is None:
        exclude_folders = ['Raw', 'Candidates', 'Found']

    images = []
    for ext in SUPPORTED_EXTENSIONS:
        for img_path in folder.glob(f"*{ext}"):
            # Skip if in excluded folders
            if not any(excluded in str(img_path) for excluded in exclude_folders):
                images.append(img_path)
        for img_path in folder.glob(f"*{ext.upper()}"):
            if not any(excluded in str(img_path) for excluded in exclude_folders):
                images.append(img_path)

    return sorted(set(images))


def progress_bar(iterable, desc: str, total: int):
    """Wrapper for progress bar (uses tqdm if available)."""
    if HAS_TQDM:
        return tqdm(iterable, desc=desc, total=total, ncols=80)
    else:
        print(f"{desc}...")
        return iterable


def main():
    parser = argparse.ArgumentParser(
        description="MeteoFinder Phase 1 - Local OpenCV Detection (NO API USAGE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sensitivity levels (1-5):
  1 = Very strict (fewer candidates, might miss faint meteors)
  2 = Strict
  3 = Balanced (default, recommended)
  4 = Sensitive
  5 = Very sensitive (more candidates, catches faint trails)

This script only performs local detection. No API calls are made.
Run phase2_verify.py separately to verify candidates with Claude Vision API.
        """
    )
    parser.add_argument("folder", help="Path to folder containing images")
    parser.add_argument(
        "-s", "--sensitivity",
        type=int,
        default=DEFAULT_SENSITIVITY,
        choices=[1, 2, 3, 4, 5],
        help=f"Pre-filter sensitivity 1-5 (default: {DEFAULT_SENSITIVITY})"
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list candidates without copying files"
    )

    args = parser.parse_args()

    # Validate folder
    folder = Path(args.folder).resolve()
    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        sys.exit(1)
    if not folder.is_dir():
        print(f"Error: Not a directory: {folder}")
        sys.exit(1)

    # Find images (excluding Raw, Candidates, Found folders)
    images = find_images(folder)
    if not images:
        print(f"No supported images found in {folder}")
        print(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        sys.exit(1)

    print(f"\nMeteoFinder - PHASE 1: Local Detection (OpenCV)")
    print(f"{'=' * 60}")
    print(f"Folder: {folder}")
    print(f"Images found: {len(images)}")
    print(f"Sensitivity: {args.sensitivity} - {get_sensitivity_description(args.sensitivity)}")
    print(f"Mode: Local OpenCV only - NO API CALLS")
    print(f"{'=' * 60}\n")

    # Create output folder
    candidates_folder = folder / CANDIDATES_FOLDER
    if not args.list_only:
        candidates_folder.mkdir(exist_ok=True)

    # Phase 1: Pre-filter with OpenCV
    print("Phase 1: Pre-filtering with OpenCV...")
    print("Filters applied:")
    print("  - Brightness check (filters dark wires)")
    print("  - Dotted line detection (filters blinking airplane lights)")
    print("  - Red light detection (filters airplane navigation lights)")
    print("  - Parallel line detection (filters multi-wing airplane lights)")
    print("  - Temporal continuity (filters satellites)")
    print("  - Cloud detection (filters multi-line images)")
    print()

    candidates = []

    for img_path in progress_bar(images, "Scanning", len(images)):
        if detect_streak(str(img_path), args.sensitivity):
            candidates.append(img_path)

    print(f"\n{'=' * 60}")
    print(f"Pre-filter complete: {len(candidates)} candidates from {len(images)} images")
    print(f"Detection rate: {len(candidates)/len(images)*100:.1f}%")
    print(f"{'=' * 60}\n")

    if not candidates:
        print("No potential meteors detected.")
        print("Try increasing sensitivity (-s 4 or -s 5) to catch fainter streaks.")
        return

    if args.list_only:
        print(f"Candidates (list-only mode):")
        for c in candidates:
            print(f"  - {c.name}")
        print(f"\nTo copy candidates, run without --list-only")
        return

    # Copy candidates to Candidates folder
    print(f"Copying {len(candidates)} candidates to {CANDIDATES_FOLDER}/ folder...")
    for img_path in candidates:
        dest = candidates_folder / img_path.name
        if not dest.exists():
            shutil.copy2(img_path, dest)

    print(f"\n{'=' * 60}")
    print("PHASE 1 COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total images scanned: {len(images)}")
    print(f"Candidates found: {len(candidates)}")
    print(f"Candidates copied to: {candidates_folder}")
    print(f"\nCandidate images:")
    for img_path in candidates:
        print(f"  - {img_path.name}")

    print(f"\n{'=' * 60}")
    print("NEXT STEPS:")
    print(f"{'=' * 60}")
    print("1. Review candidates in the Candidates/ folder")
    print("2. To verify with Claude Vision API, run:")
    print(f"   python phase2_verify.py \"{folder}\"")
    print()
    print("Note: Phase 2 will use Claude API and incur costs.")
    print(f"      Estimated cost for {len(candidates)} images: ~${len(candidates) * 0.005:.2f}")


if __name__ == "__main__":
    main()
