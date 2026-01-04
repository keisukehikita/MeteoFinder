#!/usr/bin/env python3
"""Analyze actual brightness values of meteor streaks."""

import cv2
import numpy as np
from pathlib import Path

def analyze_image_brightness(image_path):
    """Analyze brightness of the brightest lines in the image."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    # Resize
    height, width = img.shape[:2]
    max_dimension = 1500
    if max(height, width) > max_dimension:
        scale_factor = max_dimension / max(height, width)
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Try multiple Canny thresholds to detect fainter lines
    all_lines = []
    for canny_low, canny_high in [(50, 150), (80, 160), (100, 200)]:
        edges = cv2.Canny(blurred, canny_low, canny_high)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                                minLineLength=50, maxLineGap=10)
        if lines is not None:
            all_lines.extend(lines)

    if not all_lines:
        return None

    # Calculate brightness for each line
    brightness_values = []
    for line in all_lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if length < 50:
            continue

        # Sample brightness along line
        num_samples = max(10, int(length / 5))
        x_points = np.linspace(x1, x2, num_samples).astype(int)
        y_points = np.linspace(y1, y2, num_samples).astype(int)

        h, w = gray.shape
        x_points = np.clip(x_points, 0, w - 1)
        y_points = np.clip(y_points, 0, h - 1)

        line_brightness = np.mean(gray[y_points, x_points])
        max_brightness = np.max(gray[y_points, x_points])

        brightness_values.append((length, line_brightness, max_brightness))

    if not brightness_values:
        return None

    # Return the brightest long line (likely the meteor)
    brightness_values.sort(key=lambda x: (x[1], x[0]), reverse=True)
    return brightness_values[0]


selected_folder = Path("D:/20251213-geminid-meteor-3/selected")
images = sorted(selected_folder.glob("*.JPG"))

print("Brightness analysis of selected meteor images:")
print(f"{'Image':<20} {'Length':>8} {'Avg Bright':>12} {'Max Bright':>12}")
print("-" * 60)

results = []
for img_path in images:
    result = analyze_image_brightness(str(img_path))
    if result:
        length, avg_bright, max_bright = result
        results.append((img_path.name, avg_bright))
        print(f"{img_path.name:<20} {length:>8.1f} {avg_bright:>12.1f} {max_bright:>12.1f}")
    else:
        print(f"{img_path.name:<20} {'No lines detected':>40}")

if results:
    avg_brightnesses = [b for _, b in results]
    print(f"\n{'='*60}")
    print(f"Statistics:")
    print(f"  Min avg brightness: {min(avg_brightnesses):.1f}")
    print(f"  Max avg brightness: {max(avg_brightnesses):.1f}")
    print(f"  Mean avg brightness: {np.mean(avg_brightnesses):.1f}")
    print(f"  Median avg brightness: {np.median(avg_brightnesses):.1f}")
    print(f"\nCurrent thresholds:")
    print(f"  Sensitivity 1: min_brightness = 180")
    print(f"  Sensitivity 2: min_brightness = 150")
    print(f"  Sensitivity 3: min_brightness = 120")
    print(f"  Sensitivity 4: min_brightness = 100")
    print(f"  Sensitivity 5: min_brightness = 80")
