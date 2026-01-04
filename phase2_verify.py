#!/usr/bin/env python3
"""
MeteoFinder Phase 2 - Claude Vision API Verification (USES API - INCURS COSTS)

This script verifies candidates from Phase 1 using Claude Vision API.
Only run this after reviewing Phase 1 candidates to avoid unnecessary API costs.

Usage:
    python phase2_verify.py <folder_path>

Examples:
    python phase2_verify.py D:\\Photos\\NightSky
    python phase2_verify.py ./images

WARNING: This script uses the Claude Vision API and will incur costs.
         Make sure you have reviewed the candidates from Phase 1 first.
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

from config import SUPPORTED_EXTENSIONS, ANTHROPIC_API_KEY
from detector.claude_vision import verify_meteor


CANDIDATES_FOLDER = "Candidates"
FOUND_FOLDER = "Found"


def has_valid_api_key() -> bool:
    """Check if a valid API key is configured."""
    return (ANTHROPIC_API_KEY
            and ANTHROPIC_API_KEY != "your-api-key-here"
            and len(ANTHROPIC_API_KEY) > 10)


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
        description="MeteoFinder Phase 2 - Claude Vision API Verification (USES API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WARNING: This script uses the Claude Vision API and will incur costs.
         Estimated cost: ~$0.005 per image

This script verifies candidates from Phase 1 using Claude Vision API.
Make sure you have run phase1_detect.py first and reviewed the candidates.
        """
    )
    parser.add_argument("folder", help="Path to folder containing images")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be verified without making API calls"
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

    # Check for Candidates folder
    candidates_folder = folder / CANDIDATES_FOLDER
    if not candidates_folder.exists():
        print(f"Error: Candidates folder not found: {candidates_folder}")
        print(f"\nPlease run Phase 1 first:")
        print(f"  python phase1_detect.py \"{folder}\"")
        sys.exit(1)

    # Find candidate images
    candidates = find_images(candidates_folder)
    if not candidates:
        print(f"No candidate images found in {candidates_folder}")
        print(f"\nPlease run Phase 1 first:")
        print(f"  python phase1_detect.py \"{folder}\"")
        sys.exit(1)

    # Check API key
    if not has_valid_api_key():
        print("Error: No valid Anthropic API key found in config.py")
        print("\nPlease add your API key to config.py:")
        print("  ANTHROPIC_API_KEY = \"your-api-key-here\"")
        print("\nGet an API key at: https://console.anthropic.com/")
        sys.exit(1)

    estimated_cost = len(candidates) * 0.005

    print(f"\nMeteoFinder - PHASE 2: Claude Vision API Verification")
    print(f"{'=' * 60}")
    print(f"Folder: {folder}")
    print(f"Candidates to verify: {len(candidates)}")
    print(f"Estimated API cost: ~${estimated_cost:.2f}")
    print(f"Mode: {'DRY RUN (no API calls)' if args.dry_run else 'LIVE - API CALLS WILL BE MADE'}")
    print(f"{'=' * 60}\n")

    if args.dry_run:
        print("DRY RUN MODE - No API calls will be made")
        print(f"\nCandidates to verify:")
        for img_path in candidates:
            print(f"  - {img_path.name}")
        print(f"\nTo actually verify with API, run without --dry-run")
        return

    # Confirm before proceeding
    print(f"WARNING: This will make {len(candidates)} API calls")
    print(f"Estimated cost: ~${estimated_cost:.2f}")
    print()
    response = input("Do you want to proceed? (yes/no): ").strip().lower()

    if response not in ['yes', 'y']:
        print("Aborted.")
        return

    # Create Found folder
    found_folder = folder / FOUND_FOLDER
    found_folder.mkdir(exist_ok=True)

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
            dest = found_folder / img_path.name
            if not dest.exists():
                shutil.copy2(img_path, dest)

        if not HAS_TQDM:
            status = "✓ METEOR" if result["is_meteor"] else "✗ not meteor"
            print(f"  {img_path.name}: {status}")

    # Summary
    print(f"\n{'=' * 60}")
    print("PHASE 2 COMPLETE - API VERIFICATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Candidates verified: {len(candidates)}")
    print(f"Confirmed meteors: {len(confirmed_meteors)}")
    print(f"False positives filtered: {len(candidates) - len(confirmed_meteors)}")
    print(f"Accuracy: {len(confirmed_meteors)/len(candidates)*100:.1f}% true positives" if candidates else "")

    if confirmed_meteors:
        print(f"\nConfirmed meteor images copied to: {found_folder}")
        print("\nDetected meteors:")
        for img_path, result in confirmed_meteors:
            conf = result['confidence']
            desc = result['description'][:60] + "..." if len(result['description']) > 60 else result['description']
            print(f"  [{conf}] {img_path.name}")
            print(f"         {desc}")
    else:
        print("\nNo meteors confirmed by Claude Vision API.")
        print("All candidates were false positives (planes, satellites, artifacts).")

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Phase 1 candidates: {len(candidates)}")
    print(f"Phase 2 confirmed: {len(confirmed_meteors)}")
    print(f"Final results in: {found_folder}")


if __name__ == "__main__":
    main()
