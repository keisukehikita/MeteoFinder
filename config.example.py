# MeteoFinder Configuration
# Copy this file to config.py and fill in your API key

# Your Anthropic API key - get one at https://console.anthropic.com/
ANTHROPIC_API_KEY = "your-api-key-here"

# Claude model for vision analysis (Sonnet is cost-effective for this task)
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# Default pre-filter sensitivity (1-5 scale)
# 1 = Very strict (fewer API calls, might miss faint meteors)
# 3 = Balanced (recommended)
# 5 = Very sensitive (more API calls, catches faint trails)
DEFAULT_SENSITIVITY = 3

# Supported image file extensions
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# Rate limiting: delay between API calls (seconds)
API_DELAY = 1.0

# Output folder name for detected meteors
OUTPUT_FOLDER = "Found"
