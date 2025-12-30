#!/usr/bin/env python3
"""
MeteoFinder - Hybrid meteor detection using OpenCV pre-filter and Claude Vision API.

Usage:
    python main.py <folder_path> [--sensitivity N]

Examples:
    python main.py C:\\Photos\\NightSky
    python main.py ./images --sensitivity 4
    python main.py ./images -s 2
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

from config import SUPPORTED_EXTENSIONS, DEFAULT_SENSITIVITY, OUTPUT_FOLDER, ANTHROPIC_API_KEY
from detector.prefilter import detect_streak, get_sensitivity_description
from detector.claude_vision import verify_meteor


def find_images(folder: Path) -> List[Path]:
    """Find all supported image files in a folder."""
    images = []
    for ext in SUPPORTED_EXTENSIONS:
        images.extend(folder.glob(f"*{ext}"))
        images.extend(folder.glob(f"*{ext.upper()}"))
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
        description="MeteoFinder - Detect meteors in night sky photos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sensitivity levels (1-5):
  1 = Very strict (fewer API calls, might miss faint meteors)
  2 = Strict
  3 = Balanced (default, recommended)
  4 = Sensitive
  5 = Very sensitive (more API calls, catches faint trails)
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
        "--prefilter-only",
        action="store_true",
        help="Only run pre-filter (skip Claude API verification)"
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

    # Check API key
    if not args.prefilter_only and ANTHROPIC_API_KEY == "your-api-key-here":
        print("Error: Please set your ANTHROPIC_API_KEY in config.py")
        sys.exit(1)

    # Find images
    images = find_images(folder)
    if not images:
        print(f"No supported images found in {folder}")
        print(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        sys.exit(1)

    print(f"\nMeteoFinder")
    print(f"{'=' * 50}")
    print(f"Folder: {folder}")
    print(f"Images found: {len(images)}")
    print(f"Sensitivity: {args.sensitivity} - {get_sensitivity_description(args.sensitivity)}")
    print(f"{'=' * 50}\n")

    # Create output folder
    output_folder = folder / OUTPUT_FOLDER
    output_folder.mkdir(exist_ok=True)

    # Phase 1: Pre-filter with OpenCV
    print("Phase 1: Pre-filtering with OpenCV...")
    candidates = []

    for img_path in progress_bar(images, "Scanning", len(images)):
        if detect_streak(str(img_path), args.sensitivity):
            candidates.append(img_path)

    print(f"\nPre-filter complete: {len(candidates)} candidates from {len(images)} images")

    if not candidates:
        print("\nNo potential meteors detected.")
        print("Try increasing sensitivity (-s 4 or -s 5) to catch fainter streaks.")
        return

    if args.prefilter_only:
        print(f"\nPre-filter only mode. Candidates:")
        for c in candidates:
            print(f"  - {c.name}")
        return

    # Phase 2: Verify with Claude API
    print(f"\nPhase 2: Verifying {len(candidates)} candidates with Claude Vision API...")
    confirmed_meteors = []

    for img_path in progress_bar(candidates, "Verifying", len(candidates)):
        result = verify_meteor(str(img_path))

        if result.get("error"):
            print(f"\n  Warning: {img_path.name} - {result['error']}")
            continue

        if result["is_meteor"]:
            confirmed_meteors.append((img_path, result))
            # Copy to Found folder
            dest = output_folder / img_path.name
            if not dest.exists():
                shutil.copy2(img_path, dest)

        if not HAS_TQDM:
            status = "METEOR" if result["is_meteor"] else "no meteor"
            print(f"  {img_path.name}: {status}")

    # Summary
    print(f"\n{'=' * 50}")
    print("RESULTS")
    print(f"{'=' * 50}")
    print(f"Total images scanned: {len(images)}")
    print(f"Pre-filter candidates: {len(candidates)}")
    print(f"Confirmed meteors: {len(confirmed_meteors)}")

    if confirmed_meteors:
        print(f"\nMeteor images copied to: {output_folder}")
        print("\nDetected meteors:")
        for img_path, result in confirmed_meteors:
            conf = result['confidence']
            desc = result['description'][:60] + "..." if len(result['description']) > 60 else result['description']
            print(f"  [{conf}] {img_path.name}")
            print(f"         {desc}")
    else:
        print("\nNo meteors confirmed.")
        print("Tips:")
        print("  - Try increasing sensitivity (-s 4 or -s 5)")
        print("  - Check if images contain visible meteor streaks")


if __name__ == "__main__":
    main()
