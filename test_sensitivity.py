#!/usr/bin/env python3
"""Test different sensitivity levels on missed meteors."""

from pathlib import Path
from detector.prefilter import detect_streak

selected_folder = Path("D:/20251213-geminid-meteor-3/selected")
images = sorted(selected_folder.glob("*.JPG"))

print(f"Testing {len(images)} selected meteors with different sensitivity levels:\n")

for sensitivity in [1, 2, 3, 4, 5]:
    detected = 0
    for img_path in images:
        if detect_streak(str(img_path), sensitivity):
            detected += 1

    print(f"Sensitivity {sensitivity}: {detected}/{len(images)} detected ({detected/len(images)*100:.1f}%)")
