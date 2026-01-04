#!/usr/bin/env python3
"""
Diagnostic script to analyze why certain meteor images were not detected.
"""

import cv2
import numpy as np
from pathlib import Path
from detector.prefilter import (
    SENSITIVITY_PARAMS,
    _check_line_brightness,
    _is_dotted_line,
    _has_red_lights_on_streak,
    _get_streak_position,
    _count_parallel_lines,
    _are_images_adjacent,
    _streaks_are_similar
)

def diagnose_image(image_path: str, sensitivity: int = 1):
    """Diagnose why an image was not detected as a meteor."""

    print(f"\n{'='*80}")
    print(f"Analyzing: {Path(image_path).name}")
    print(f"{'='*80}")

    sensitivity = max(1, min(5, sensitivity))
    params = SENSITIVITY_PARAMS[sensitivity]
    canny_low, canny_high, min_line_length, max_line_gap, min_brightness = params

    print(f"Sensitivity: {sensitivity}")
    print(f"Parameters: canny_low={canny_low}, canny_high={canny_high}, min_line_length={min_line_length}, max_line_gap={max_line_gap}, min_brightness={min_brightness}")

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"ERROR: Could not load image")
        return

    original_img = img.copy()

    # Resize if needed
    height, width = img.shape[:2]
    max_dimension = 1500
    scale_factor = 1.0
    if max(height, width) > max_dimension:
        scale_factor = max_dimension / max(height, width)
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
        min_line_length = int(min_line_length * scale_factor)

    print(f"Image size: {width}x{height}, scale_factor: {scale_factor:.2f}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # Detect lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    if lines is None:
        print("REJECTED: No lines detected by Hough transform")
        return

    print(f"Lines detected by Hough: {len(lines)}")

    # Analyze each line
    current_streaks = []
    rejection_reasons = []

    for idx, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        reason = None

        # Check length
        if length < min_line_length:
            reason = f"too short ({length:.1f} < {min_line_length})"

        # Check brightness
        elif not _check_line_brightness(gray, x1, y1, x2, y2, min_brightness):
            reason = "failed brightness check (too dim or darker than background)"

        # Check angle
        else:
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if not (5 < angle < 175 and angle != 90):
                reason = f"invalid angle ({angle:.1f}°) - too horizontal/vertical"

        # Check dotted pattern
            elif _is_dotted_line(gray, x1, y1, x2, y2, min_brightness):
                reason = "dotted/dashed line pattern (blinking airplane lights)"

        # Check red lights
            else:
                if scale_factor != 1.0:
                    ox1, oy1 = int(x1 / scale_factor), int(y1 / scale_factor)
                    ox2, oy2 = int(x2 / scale_factor), int(y2 / scale_factor)
                else:
                    ox1, oy1, ox2, oy2 = x1, y1, x2, y2

                if _has_red_lights_on_streak(original_img, ox1, oy1, ox2, oy2):
                    reason = "red navigation lights detected (airplane)"
                else:
                    # Line passed all individual checks
                    streak_pos = _get_streak_position(x1, y1, x2, y2, img.shape[:2])
                    current_streaks.append(streak_pos)

        if reason:
            rejection_reasons.append((idx, length, reason))

    print(f"\nLine-level filtering:")
    print(f"  Passed individual checks: {len(current_streaks)}/{len(lines)}")

    if rejection_reasons:
        print(f"\n  Sample rejections (first 5):")
        for idx, length, reason in rejection_reasons[:5]:
            print(f"    Line {idx}: length={length:.1f}px - {reason}")

    # Check image-level filters
    if not current_streaks:
        print("\nREJECTED: No lines passed individual filtering")
        return

    # Check parallel lines
    parallel_count = _count_parallel_lines(current_streaks)
    if parallel_count >= 2:
        print(f"\nREJECTED: Parallel lines detected (count={parallel_count}) - multi-wing airplane")
        return

    # Check cloud filter
    if len(current_streaks) > 4:
        print(f"\nREJECTED: Too many lines ({len(current_streaks)} > 4) - likely clouds")
        return

    # Temporal continuity check (simplified - we can't check adjacent images here)
    print(f"\nPASSED all filters! {len(current_streaks)} streak(s) detected")
    print(f"  (Note: Temporal continuity check skipped in diagnostic mode)")


def main():
    import sys

    selected_folder = Path("D:/20251213-geminid-meteor-3/selected")

    # Get all JPG files
    images = sorted(selected_folder.glob("*.JPG"))

    print(f"Analyzing {len(images)} selected meteor images...")
    print(f"These are meteors you identified that the app missed.\n")

    # Test with different sensitivity levels
    for sensitivity in [1, 2, 3]:
        print(f"\n{'#'*80}")
        print(f"# TESTING WITH SENSITIVITY = {sensitivity}")
        print(f"{'#'*80}")

        for img_path in images[:5]:  # Test first 5 for now
            diagnose_image(str(img_path), sensitivity)

        if sensitivity == 1:
            break  # Start with sensitivity 1 only


if __name__ == "__main__":
    main()
