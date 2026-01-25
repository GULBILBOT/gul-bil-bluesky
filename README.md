# Yellow Car Bot 🚕

An automated bot that monitors Norwegian traffic cameras for yellow cars and posts them to Bluesky with the classic Norwegian phrase "GUL BIL!" (Yellow Car!).

## How It Works

1. **Downloads images** from 814 Norwegian traffic cameras
2. **Detects vehicles** using YOLO26 real-time object detection
3. **Validates yellow color** using HSV filtering for accurate yellow car detection
4. **Posts to Bluesky** with "GUL BIL!" and annotated bounding boxes when a yellow car is found

> **Note**: This bot has been migrated from GPT-4o/OWLv2 (cloud-based AI) to **YOLO26** (local inference) for improved speed, cost-efficiency, and reliability. YOLO26 provides ~50x faster detection (100-200ms vs 5-10s per image) while maintaining high accuracy.

## Features

- **Fair Processing**: Shuffles camera order to ensure all cameras get processed at different times of day
- **Budget Optimized**: Designed to stay within GitHub Actions' 2000 minutes/month limit
- **Rate Limit Aware**: Gracefully handles API rate limiting
- **Persistent State**: Remembers progress between runs
- **Statistics Tracking**: Monitors detection rates and posting success

## Setup

### 1. Repository Structure
```
your-repo/
├── gulbilbot.py
├── pyproject.toml
├── valid_webcam_ids.txt
├── .github/
│   └── workflows/
│       └── post.yml
└── README.md
```

### 2. Required Files

**pyproject.toml**:
```txt
requests
python-dotenv
Pillow
atproto
```

**valid_webcam_ids.txt**:
One traffic camera URL per line (814 URLs total). Example:
```
https://webkamera.atlas.vegvesen.no/public/kamera?id=3000063_1
https://webkamera.atlas.vegvesen.no/public/kamera?id=3001004_1
...
```

### 3. GitHub Secrets

Set up these secrets in your repository (Settings → Secrets and variables → Actions):

- `BSKY_HANDLE`: Your Bluesky handle (e.g., `username.bsky.social`)
- `BSKY_PASSWORD`: Your Bluesky app password (not your main password!)

Note: No API tokens required! YOLO26 runs locally on GitHub Actions runners.

### 4. Bluesky App Password

1. Go to Settings → Privacy and Security → App Passwords
2. Create a new app password for this bot
3. Use this password (not your main password) in the `BSKY_PASSWORD` secret

## Configuration

Edit these variables in `gulbilbot.py`:

```python
CONF_THRESHOLD = 0.3          # YOLO26 vehicle detection confidence (lower = more sensitive)
YELLOW_RATIO_THRESHOLD = 0.15 # Minimum ratio of yellow pixels to confirm yellow car
SAMPLE_SIZE = 100             # Number of images to test per run
```

**Supported Vehicle Types**: car, truck, bus, van, threewheel

## Schedule

The bot runs **3 times per day**:
- 6:00 AM UTC
- 2:00 PM UTC
- 10:00 PM UTC

**Budget**: ~60 minutes/day, ~1800 minutes/month (within 2000 limit)

**Coverage**: 300 random cameras per run, all 814 cameras covered every ~3 runs

## Statistics

Each run tracks:
- Images downloaded successfully
- Yellow cars detected
- Successful Bluesky posts
- Processing time
- Results saved to `test_results.txt`

## Yellow Detection Algorithm

**Two-stage process**:

1. **YOLO26 Object Detection**:
    - Detects vehicles in real-time using YOLO26n (Nano variant)
    - Identifies: cars, trucks, buses, vans, and threewheels
    - Confidence threshold: 0.3 (30%)
    - Processing speed: 100-200ms per image
    - ~50x faster than previous cloud-based approach

2. **HSV Color Validation**:
    - Validates detected vehicle contains sufficient yellow pixels
    - HSV Range: Hue=[15-35], Saturation=[80-255], Value=[80-255]
    - Minimum yellow pixel ratio: 15%
    - Eliminates false positives from non-yellow vehicles

3. **Bounding Box Annotation**:
    - Draws bright yellow rectangles around detected yellow cars
    - Labels include vehicle type and confidence score
    - Annotated images posted directly to Bluesky

## Troubleshooting

### Common Issues

**"Rate limit reached"**:
- Bot automatically stops to preserve GitHub Actions minutes
- Will resume in next scheduled run

**"Shuffle state not found"**:
- Normal on first run
- Bot will create new shuffle and start processing

**"No yellow cluster detected"**:
- Most images won't have yellow cars - this is normal
- Adjust `YELLOW_THRESHOLD` if too sensitive/insensitive

### Logs

Check workflow logs for:
- Processing progress
- Detection statistics
- Error messages
- Session summaries

### Manual Testing

Trigger a manual run:
1. Go to Actions tab
2. Select "Yellow Car Bot - Budget Optimized"
3. Click "Run workflow"

## File Descriptions

- **`gulbilbot.py`**: Main bot script (downloads 300 random images, detects yellow cars, posts to Bluesky)
- **`pyproject.toml`**: Python dependencies
- **`valid_webcam_ids.txt`**: List of 814 traffic camera URLs
- **`.github/workflows/post.yml`**: GitHub Actions workflow (runs 3x daily)
- **`test_results.txt`**: Results from latest run (auto-generated)

## Privacy & Ethics

- Only processes public traffic camera feeds
- No personal data collected or stored
- Images deleted immediately after processing
- Respects API rate limits and terms of service

## Contributing

Feel free to:
- Adjust detection parameters for better accuracy
- Add more traffic cameras to the list
- Improve the yellow detection algorithm
- Enhance error handling

## License

This project is for educational and entertainment purposes. Please respect:
- Traffic camera usage terms
- Bluesky community guidelines
- API rate limits and quotas

---

**Lykke til med GUL BIL-jakten!** 🚕