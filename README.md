# MeteoFinder

A hybrid meteor detection tool that uses OpenCV pre-filtering and Claude Vision API to find meteors in night sky photographs.

## How It Works

1. **Pre-filter (OpenCV)**: Quickly scans images for linear streaks using edge detection and Hough transforms
2. **Verify (Claude API)**: Sends candidates to Claude Vision API to confirm meteor presence and filter out airplanes, satellites, star trails, etc.

This hybrid approach reduces API costs by ~85% compared to sending all images to the API.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MeteoFinder.git
cd MeteoFinder

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp config.example.py config.py
# Edit config.py and add your Anthropic API key
```

## Usage

```bash
# Basic usage
python main.py /path/to/your/photos

# Adjust pre-filter sensitivity (1-5, default: 3)
python main.py /path/to/photos --sensitivity 4

# Pre-filter only (no API calls, for testing)
python main.py /path/to/photos --prefilter-only
```

### Sensitivity Levels

| Level | Description | Estimated API Calls |
|-------|-------------|---------------------|
| 1 | Very strict - only obvious streaks | ~5% of images |
| 2 | Strict | ~10% |
| 3 | Balanced (recommended) | ~15% |
| 4 | Sensitive | ~25% |
| 5 | Very sensitive - catches faint trails | ~35%+ |

## Output

Detected meteor images are copied to a `Found/` subfolder within the scanned directory.

```
YourPhotos/
├── IMG_001.jpg
├── IMG_002.jpg
├── ...
└── Found/           # Meteor images copied here
    └── IMG_042.jpg
```

## Cost Estimate

For 500 images per night with sensitivity 3:
- ~75 images sent to API (15% pass pre-filter)
- Cost: ~$0.50/night

## Requirements

- Python 3.10+
- Anthropic API key ([get one here](https://console.anthropic.com/))

## License

MIT
