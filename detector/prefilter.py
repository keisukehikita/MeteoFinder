"""
OpenCV-based pre-filter for meteor detection.
Detects linear streaks in night sky images using edge detection and Hough transforms.
"""

import cv2
import numpy as np
from pathlib import Path


# Sensitivity parameters mapping (1-5 scale)
# Each tuple: (canny_low, canny_high, min_line_length, max_line_gap, min_brightness)
SENSITIVITY_PARAMS = {
    1: (100, 200, 100, 5, 180),   # Very strict
    2: (80, 160, 80, 8, 150),     # Strict
    3: (50, 150, 50, 10, 120),    # Balanced
    4: (30, 100, 30, 15, 100),    # Sensitive
    5: (20, 80, 20, 20, 80),      # Very sensitive
}


def detect_streak(image_path: str, sensitivity: int = 3) -> bool:
    """
    Detect if an image contains a potential meteor streak.

    Args:
        image_path: Path to the image file
        sensitivity: Detection sensitivity 1-5 (default 3)
                    1 = Very strict, 5 = Very sensitive

    Returns:
        True if a potential meteor streak is detected, False otherwise
    """
    sensitivity = max(1, min(5, sensitivity))
    params = SENSITIVITY_PARAMS[sensitivity]
    canny_low, canny_high, min_line_length, max_line_gap, min_brightness = params

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not load image {image_path}")
        return False

    # Resize for faster processing if image is very large
    height, width = img.shape[:2]
    max_dimension = 1500
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        # Adjust line length threshold for scaled image
        min_line_length = int(min_line_length * scale)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    if lines is None:
        return False

    # Filter lines based on meteor characteristics
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calculate line length
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Skip short lines
        if length < min_line_length:
            continue

        # Check brightness along the line
        if _check_line_brightness(gray, x1, y1, x2, y2, min_brightness):
            # Additional check: meteors are usually not perfectly horizontal/vertical
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            # Exclude nearly horizontal lines (likely horizon artifacts)
            if 5 < angle < 175 and angle != 90:
                return True

    return False


def _check_line_brightness(gray_img: np.ndarray, x1: int, y1: int,
                           x2: int, y2: int, min_brightness: int) -> bool:
    """
    Check if the pixels along a line are bright enough to be a meteor.

    Args:
        gray_img: Grayscale image
        x1, y1, x2, y2: Line coordinates
        min_brightness: Minimum average brightness threshold

    Returns:
        True if the line is bright enough
    """
    # Sample points along the line
    num_samples = max(10, int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 5))

    x_points = np.linspace(x1, x2, num_samples).astype(int)
    y_points = np.linspace(y1, y2, num_samples).astype(int)

    # Ensure points are within image bounds
    height, width = gray_img.shape
    x_points = np.clip(x_points, 0, width - 1)
    y_points = np.clip(y_points, 0, height - 1)

    # Get brightness values along the line
    brightness_values = gray_img[y_points, x_points]

    # Check if average brightness exceeds threshold
    avg_brightness = np.mean(brightness_values)

    return avg_brightness >= min_brightness


def get_sensitivity_description(sensitivity: int) -> str:
    """Get a human-readable description of the sensitivity level."""
    descriptions = {
        1: "Very strict (fewer candidates, might miss faint meteors)",
        2: "Strict",
        3: "Balanced (recommended)",
        4: "Sensitive",
        5: "Very sensitive (more candidates, catches faint trails)"
    }
    return descriptions.get(sensitivity, "Unknown")
