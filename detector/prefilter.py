"""
OpenCV-based pre-filter for meteor detection.
Detects linear streaks in night sky images using edge detection and Hough transforms.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List


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
    Also checks that the line is brighter than its surroundings (not a dark wire).

    Args:
        gray_img: Grayscale image
        x1, y1, x2, y2: Line coordinates
        min_brightness: Minimum average brightness threshold

    Returns:
        True if the line is bright enough and brighter than background
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

    if avg_brightness < min_brightness:
        return False

    # NEW: Check that line is brighter than surrounding background
    # Sample perpendicular to the line to get background brightness
    # Calculate perpendicular direction
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx * dx + dy * dy)

    if length == 0:
        return False

    # Normalized perpendicular vector
    perp_x = -dy / length
    perp_y = dx / length

    # Sample background on both sides of the line (offset by 10-20 pixels)
    background_samples = []
    offset_distances = [10, 15, 20]

    for offset in offset_distances:
        for direction in [-1, 1]:  # Both sides
            # Sample a few points along the line with perpendicular offset
            for i in [0.25, 0.5, 0.75]:  # Sample at 25%, 50%, 75% along line
                sample_x = int(x1 + i * dx + direction * offset * perp_x)
                sample_y = int(y1 + i * dy + direction * offset * perp_y)

                # Check bounds
                if 0 <= sample_x < width and 0 <= sample_y < height:
                    background_samples.append(gray_img[sample_y, sample_x])

    if not background_samples:
        return True  # Can't determine background, assume OK

    avg_background = np.mean(background_samples)

    # Meteor should be BRIGHTER than background (not darker like wires)
    # Require line to be at least 20% brighter than background
    return avg_brightness > avg_background * 1.2


def _has_red_lights_on_streak(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
    """
    Check if there are red dots along the streak (airplane navigation lights).

    Args:
        img: BGR color image
        x1, y1, x2, y2: Line coordinates

    Returns:
        True if red lights detected (likely airplane)
    """
    # Sample points along the line
    num_samples = max(20, int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 3))

    x_points = np.linspace(x1, x2, num_samples).astype(int)
    y_points = np.linspace(y1, y2, num_samples).astype(int)

    # Ensure points are within image bounds
    height, width = img.shape[:2]
    x_points = np.clip(x_points, 0, width - 1)
    y_points = np.clip(y_points, 0, height - 1)

    red_count = 0

    # Check each point for red color (higher red than green and blue)
    for x, y in zip(x_points, y_points):
        # Sample a small region around the point (3x3)
        y_min, y_max = max(0, y-1), min(height, y+2)
        x_min, x_max = max(0, x-1), min(width, x+2)

        region = img[y_min:y_max, x_min:x_max]

        # BGR format - check if red channel is dominant
        mean_color = np.mean(region, axis=(0, 1))
        b, g, r = mean_color

        # Red light detection: red channel significantly higher than others
        if r > 100 and r > g * 1.5 and r > b * 1.5:
            red_count += 1

    # If we detect multiple red dots along the streak, it's likely an airplane
    return red_count >= 2


def _get_streak_position(x1: int, y1: int, x2: int, y2: int,
                         img_shape: Tuple[int, int]) -> Tuple[float, float, float]:
    """
    Get normalized position and angle of a streak for comparison.

    Returns:
        (normalized_center_x, normalized_center_y, angle_degrees)
    """
    height, width = img_shape

    center_x = (x1 + x2) / 2 / width
    center_y = (y1 + y2) / 2 / height
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

    return (center_x, center_y, angle)


def _streaks_are_similar(pos1: Tuple[float, float, float],
                          pos2: Tuple[float, float, float],
                          tolerance: float = 0.15) -> bool:
    """
    Check if two streak positions are similar (indicating airplane/satellite).

    Args:
        pos1, pos2: Streak positions (center_x, center_y, angle)
        tolerance: Position tolerance (0-1 scale)
    """
    x1, y1, a1 = pos1
    x2, y2, a2 = pos2

    # Check position similarity
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Check angle similarity (normalize to 0-180 range)
    angle_diff = abs(a1 - a2)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff

    # Similar position and angle = likely same object (airplane/satellite)
    return distance < tolerance and angle_diff < 20


# Cache for previous image streak data to compare temporal continuity
_previous_streak_cache: List[Tuple[str, Tuple[float, float, float]]] = []


def detect_streak(image_path: str, sensitivity: int = 3) -> bool:
    """
    Detect if an image contains a potential meteor streak.

    Filters out airplanes by:
    - Detecting red navigation lights along streaks
    - Checking for streaks in similar positions across consecutive images

    Args:
        image_path: Path to the image file
        sensitivity: Detection sensitivity 1-5 (default 3)
                    1 = Very strict, 5 = Very sensitive

    Returns:
        True if a potential meteor streak is detected, False otherwise
    """
    global _previous_streak_cache

    sensitivity = max(1, min(5, sensitivity))
    params = SENSITIVITY_PARAMS[sensitivity]
    canny_low, canny_high, min_line_length, max_line_gap, min_brightness = params

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not load image {image_path}")
        return False

    original_img = img.copy()  # Keep original for red light detection

    # Resize for faster processing if image is very large
    height, width = img.shape[:2]
    max_dimension = 1500
    scale_factor = 1.0
    if max(height, width) > max_dimension:
        scale_factor = max_dimension / max(height, width)
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
        # Adjust line length threshold for scaled image
        min_line_length = int(min_line_length * scale_factor)

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
    path_obj = Path(image_path)
    current_streaks = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calculate line length
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Skip short lines
        if length < min_line_length:
            continue

        # Check brightness along the line
        if not _check_line_brightness(gray, x1, y1, x2, y2, min_brightness):
            continue

        # Additional check: meteors are usually not perfectly horizontal/vertical
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        # Exclude nearly horizontal lines (likely horizon artifacts)
        if not (5 < angle < 175 and angle != 90):
            continue

        # NEW: Check for red lights along the streak (airplane navigation lights)
        # Scale coordinates back to original image if needed
        if scale_factor != 1.0:
            ox1, oy1 = int(x1 / scale_factor), int(y1 / scale_factor)
            ox2, oy2 = int(x2 / scale_factor), int(y2 / scale_factor)
        else:
            ox1, oy1, ox2, oy2 = x1, y1, x2, y2

        if _has_red_lights_on_streak(original_img, ox1, oy1, ox2, oy2):
            # Red lights detected - likely an airplane
            continue

        # Store streak position for temporal comparison
        streak_pos = _get_streak_position(x1, y1, x2, y2, img.shape[:2])
        current_streaks.append(streak_pos)

    if not current_streaks:
        return False

    # NEW: Filter out clouds - clouds create many random lines
    # Meteors are typically isolated (1-3 streaks max per image)
    # More than 4 lines usually indicates clouds, star trails, or other artifacts
    if len(current_streaks) > 4:
        return False

    # NEW: Check for temporal continuity with previous/next images
    # If a streak appears in similar position in consecutive images, it's likely airplane/satellite
    has_meteor_candidate = False

    for current_pos in current_streaks:
        is_continuous = False

        # Compare with cached previous images
        for prev_path, prev_pos in _previous_streak_cache:
            # Only compare with adjacent images (similar filenames)
            if _are_images_adjacent(path_obj.name, Path(prev_path).name):
                if _streaks_are_similar(current_pos, prev_pos):
                    is_continuous = True
                    break

        if not is_continuous:
            # Found an isolated streak (not continuing from previous images)
            has_meteor_candidate = True
            break

    # Update cache with current image streaks (keep last 3 images)
    _previous_streak_cache = [(image_path, pos) for pos in current_streaks] + _previous_streak_cache[:6]

    return has_meteor_candidate


def _are_images_adjacent(name1: str, name2: str) -> bool:
    """
    Check if two image filenames are from consecutive shots.
    Assumes common naming patterns like DSC_1234.jpg, IMG_5678.jpg, etc.
    """
    import re

    # Extract numbers from filenames
    nums1 = re.findall(r'\d+', name1)
    nums2 = re.findall(r'\d+', name2)

    if not nums1 or not nums2:
        return False

    # Compare the last number (usually the sequence number)
    try:
        num1 = int(nums1[-1])
        num2 = int(nums2[-1])

        # Consider adjacent if within 2 frames
        return abs(num1 - num2) <= 2
    except (ValueError, IndexError):
        return False


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
