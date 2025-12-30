"""
Claude Vision API integration for meteor verification.
Uses Claude's vision capabilities to confirm meteor presence in images.
"""

import base64
import io
import json
import time
from pathlib import Path
from typing import Optional

import anthropic
import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, API_DELAY


# Track last API call time for rate limiting
_last_call_time = 0.0

# Maximum RAW file size before resizing
# Claude API limit is 5MB for base64 encoded image
# Base64 adds ~33% overhead, so 3.5MB raw -> ~4.7MB base64
MAX_RAW_BYTES = 3_500_000


METEOR_DETECTION_PROMPT = """Analyze this night sky photograph. Is there a meteor (shooting star) visible in the image?

A meteor appears as:
- A bright, straight streak of light across the sky
- Usually white, yellow, or slightly colored
- Can be faint or bright
- Has a defined start and end point (not extending to edges)

Do NOT confuse meteors with:
- Airplane trails (have gaps, blinking lights, or are very long/straight)
- Satellites (steady, continuous light moving slowly)
- Star trails (curved arcs from long exposure)
- Lens flares (usually near bright objects, have specific patterns)
- Scratches or artifacts (irregular, not luminous)

Respond ONLY with valid JSON in this exact format:
{"is_meteor": true, "confidence": "high", "description": "brief description"}

Where:
- is_meteor: true if a meteor is visible, false otherwise
- confidence: "high", "medium", or "low"
- description: brief explanation of what you see (1-2 sentences)"""


def verify_meteor(image_path: str) -> dict:
    """
    Use Claude Vision API to verify if an image contains a meteor.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary with keys:
        - is_meteor: bool
        - confidence: str ("high", "medium", "low")
        - description: str
        - error: str (only if an error occurred)
    """
    global _last_call_time

    # Rate limiting
    elapsed = time.time() - _last_call_time
    if elapsed < API_DELAY:
        time.sleep(API_DELAY - elapsed)

    try:
        # Read and encode image (with automatic resizing if needed)
        image_data, media_type = _load_image_as_base64(image_path)
        if image_data is None:
            return {
                "is_meteor": False,
                "confidence": "low",
                "description": "Failed to load image",
                "error": "Could not read image file"
            }

        # Initialize Anthropic client
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        # Make API call
        _last_call_time = time.time()

        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": METEOR_DETECTION_PROMPT
                        }
                    ],
                }
            ],
        )

        # Parse response
        response_text = message.content[0].text.strip()
        return _parse_response(response_text)

    except anthropic.APIError as e:
        return {
            "is_meteor": False,
            "confidence": "low",
            "description": f"API error: {str(e)}",
            "error": str(e)
        }
    except Exception as e:
        return {
            "is_meteor": False,
            "confidence": "low",
            "description": f"Error: {str(e)}",
            "error": str(e)
        }


def _load_image_as_base64(image_path: str) -> tuple[Optional[str], str]:
    """Load an image file and return as base64 string, resizing if too large.

    Returns:
        Tuple of (base64_data, media_type) or (None, "") on error.
    """
    try:
        # First, try loading the raw file
        with open(image_path, "rb") as f:
            raw_data = f.read()

        # If file is small enough, use it directly
        if len(raw_data) <= MAX_RAW_BYTES:
            media_type = _get_media_type(image_path)
            return base64.standard_b64encode(raw_data).decode("utf-8"), media_type

        # File too large - resize with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return None, ""

        # Calculate resize factor to get under size limit
        # Start with a reasonable max dimension
        height, width = img.shape[:2]
        max_dim = 2000  # Start with 2000px max dimension

        while True:
            # Resize if needed
            if max(height, width) > max_dim:
                scale = max_dim / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            else:
                resized = img

            # Encode as JPEG with quality adjustment
            quality = 85
            while quality >= 50:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                _, encoded = cv2.imencode('.jpg', resized, encode_param)
                encoded_bytes = encoded.tobytes()

                if len(encoded_bytes) <= MAX_RAW_BYTES:
                    return base64.standard_b64encode(encoded_bytes).decode("utf-8"), "image/jpeg"

                quality -= 10

            # If still too large, reduce max dimension
            max_dim -= 200
            if max_dim < 800:
                # Last resort: very aggressive compression
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                _, encoded = cv2.imencode('.jpg', resized, encode_param)
                return base64.standard_b64encode(encoded.tobytes()).decode("utf-8"), "image/jpeg"

    except Exception:
        return None, ""


def _get_media_type(image_path: str) -> str:
    """Get the MIME type for an image file."""
    ext = Path(image_path).suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }
    return media_types.get(ext, "image/jpeg")


def _parse_response(response_text: str) -> dict:
    """Parse Claude's JSON response."""
    try:
        # Try to extract JSON from response
        # Handle case where response might have extra text
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start != -1 and end > start:
            json_str = response_text[start:end]
            result = json.loads(json_str)

            return {
                "is_meteor": bool(result.get("is_meteor", False)),
                "confidence": str(result.get("confidence", "low")),
                "description": str(result.get("description", ""))
            }
    except json.JSONDecodeError:
        pass

    # Fallback: try to interpret the response
    response_lower = response_text.lower()
    is_meteor = "meteor" in response_lower and ("yes" in response_lower or "true" in response_lower or "is a meteor" in response_lower)

    return {
        "is_meteor": is_meteor,
        "confidence": "low",
        "description": response_text[:200]
    }


def test_api_connection() -> bool:
    """Test if the API key is valid and connection works."""
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        # Simple test call
        client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}]
        )
        return True
    except Exception:
        return False
