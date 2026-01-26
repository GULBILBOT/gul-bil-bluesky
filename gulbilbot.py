#!/usr/bin/env python3
"""
GUL BIL Bot - Detects yellow cars on Norwegian traffic cameras and posts to Bluesky
Downloads 300 random webcam images, detects yellow vehicles using YOLO26, and posts them
"""

import base64
import io
import os
import random
import requests
import logging
from pathlib import Path
from datetime import datetime, timezone
from atproto import Client, models
from dotenv import load_dotenv
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

TEST_IMAGES_FOLDER = Path("test_images_folder")
TEST_IMAGES_FOLDER.mkdir(exist_ok=True)

# Read webcam URLs
WEBCAM_URLS_FILE = Path("valid_webcam_ids.txt")

# Bluesky credentials
BSKY_HANDLE = os.getenv("BSKY_HANDLE")
BSKY_PASSWORD = os.getenv("BSKY_PASSWORD")

# YOLO26 configuration - BALANCED DETECTION
YOLO_MODEL_PATH = "yolo26n.pt"
CONF_THRESHOLD = 0.3  # Lowered back to 0.3 to catch vehicles like yellow buses/taxis
YELLOW_RATIO_THRESHOLD = 0.35  # Lowered to 0.35 to catch yellow vehicles with varied shades

# Global YOLO model
yolo_model = None


def load_yolo_model():
    """Load YOLO26 model once at startup"""
    global yolo_model
    
    if yolo_model is not None:
        return True
    
    try:
        logging.info(f"Loading YOLO26 model from {YOLO_MODEL_PATH}...")
        yolo_model = YOLO(YOLO_MODEL_PATH)
        logging.info("✅ YOLO26 model loaded successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to load YOLO26 model: {e}")
        return False


def detect_yellow_car(image_path):
    """
    Run YOLO26 on an image and check for 'car' detections with yellow color.
    Returns dict with detection info: {'detected': bool, 'boxes': [(x1, y1, x2, y2, class_name, conf, yellow_ratio)]}
    
    Optimized flow:
    1. Run YOLO26 detection
    2. Skip if no vehicles found
    3. Only check for yellow color on relevant vehicle types
    """
    global yolo_model
    
    if yolo_model is None:
        if not load_yolo_model():
            return {"detected": False, "boxes": []}
    
    try:
        # Load image with OpenCV
        img = cv2.imread(str(image_path))
        if img is None:
            if image_path.exists():
                file_size = image_path.stat().st_size
                with open(image_path, "rb") as f:
                    header = f.read(16)
                logging.error(f"Could not load image: {image_path} (size={file_size}, header={header[:8].hex()})")
            else:
                logging.error(f"Could not load image: {image_path} (file does not exist)")
            return {"detected": False, "boxes": []}

        # Run YOLO26 inference
        results = yolo_model(img, verbose=False)
        yellow_boxes = []
        
        # Supported vehicle types for yellow car detection
        VEHICLE_TYPES = {"car", "truck", "bus", "van", "threewheel"}

        # Parse detections
        for res in results:
            # Check if there are ANY detections at all
            if len(res.boxes.data) == 0:
                continue
            
            for det in res.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls_id = det

                # Skip low-confidence detections
                if conf < CONF_THRESHOLD:
                    continue

                # Check if detected class is a supported vehicle type
                class_name = yolo_model.names[int(cls_id)]
                if class_name not in VEHICLE_TYPES:
                    continue

                # Only now do the expensive color check - crop the bounding box region
                x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
                
                # Ensure valid crop coordinates
                h, w = img.shape[:2]
                x1i, y1i = max(0, x1i), max(0, y1i)
                x2i, y2i = min(w, x2i), min(h, y2i)
                
                if x2i <= x1i or y2i <= y1i:
                    continue
                
                crop = img[y1i:y2i, x1i:x2i]

                # Convert to HSV and count yellow pixels
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                
                # HSV range for yellow (balanced to catch various yellow vehicles)
                # Hue: 15-35 (yellow range, includes some orange/lime shades)
                # Saturation: 80-255 (includes both bright and muted yellows)
                # Value: 80-255 (includes bright yellows and slightly darker tones)
                lower_yellow = np.array([15, 80, 80])
                upper_yellow = np.array([35, 255, 255])
                mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

                # Calculate yellow ratio in crop
                yellow_pixels = (mask > 0).sum()
                total_pixels = mask.size
                yellow_ratio = yellow_pixels / (total_pixels + 1e-6)
                
                logging.debug(f"Detected {class_name} (conf={conf:.2f}): yellow_ratio={yellow_ratio:.3f}")
                
                if yellow_ratio > YELLOW_RATIO_THRESHOLD:
                    logging.info(f"🟡 Yellow {class_name} detected!")
                    logging.info(f"   Confidence: {conf:.3f}")
                    logging.info(f"   Yellow ratio: {yellow_ratio:.3f}")
                    logging.info(f"   Bounding box: ({x1i},{y1i}) to ({x2i},{y2i})")
                    yellow_boxes.append((x1i, y1i, x2i, y2i, class_name, conf, yellow_ratio))

        if yellow_boxes:
            return {"detected": True, "boxes": yellow_boxes}
        else:
            return {"detected": False, "boxes": []}
        
    except Exception as e:
        logging.error(f"Error in YOLO26 detection: {e}")
        return {"detected": False, "boxes": []}


def download_image(url, dest, timeout=10):
    """Download image from URL and validate it's a real image"""
    try:
        resp = requests.get(url, allow_redirects=True, timeout=timeout, stream=True)
        if resp.status_code != 200:
            logging.debug(f"Failed to download {url}: Status {resp.status_code}")
            return False
        
        if not resp.content or len(resp.content) == 0:
            logging.debug(f"Empty response from {url}")
            return False
        
        content_length = resp.headers.get('content-length')
        if content_length and int(content_length) == 0:
            logging.debug(f"Content-Length is 0 for {url}")
            return False
        
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest, "wb") as f:
            f.write(resp.content)
        
        if not dest.exists():
            logging.debug(f"File not created: {dest}")
            return False
        
        file_size = dest.stat().st_size
        if file_size == 0:
            logging.debug(f"File is empty after write: {dest}")
            dest.unlink()
            return False
        
        with open(dest, "rb") as f:
            header = f.read(16)
        
        # Check for valid image magic bytes
        is_jpeg = header[:2] == b'\xff\xd8'
        is_png = header[:4] == b'\x89PNG'
        is_gif = header[:4] == b'GIF8'
        is_bmp = header[:2] == b'BM'
        is_webp = header[8:12] == b'WEBP'
        is_valid_image = is_jpeg or is_png or is_gif or is_bmp or is_webp
        
        if is_valid_image:
            logging.debug(f"Downloaded {dest.name}: {file_size} bytes, valid image")
            return True
        else:
            logging.info(f"Invalid image format for {dest.name}: header {header[:16].hex()} - likely HTML/error response")
            dest.unlink()
            return False
        
    except Exception as e:
        logging.debug(f"Exception downloading {url}: {e}")
        if dest.exists():
            try:
                dest.unlink()
            except:
                pass
        return False


def draw_bounding_boxes(image_path, boxes, output_path=None):
    """
    Draw bounding boxes on detected yellow vehicles.
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Draw bounding boxes
        for x1, y1, x2, y2, class_name, conf, yellow_ratio in boxes:
            # Draw yellow rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
            
            # Draw label with class name and confidence
            label = f"{class_name} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width + 5, y1), (0, 255, 255), -1)
            
            # Draw text
            cv2.putText(img, label, (x1 + 2, y1 - 5), font, font_scale, (0, 0, 0), thickness)
        
        if output_path is None:
            output_path = image_path.parent / f"annotated_{image_path.name}"
        
        cv2.imwrite(str(output_path), img)
        logging.debug(f"Saved annotated image to {output_path}")
        return output_path
        
    except Exception as e:
        logging.error(f"Error drawing bounding boxes: {e}")
        return None


def get_image_data_url(image_file, image_format, max_size=(800, 600), quality=85):
    """Get base64 data URL for image without resizing to preserve quality for analysis"""
    try:
        with Image.open(image_file) as img:
            img = img.convert('RGB')
            
            img_bytes = io.BytesIO()
            # Use high quality to preserve image details
            img.save(img_bytes, format='JPEG', quality=95, optimize=False)
            img_bytes.seek(0)
            
            image_base64 = base64.b64encode(img_bytes.getvalue()).decode()
            
            final_size_kb = len(image_base64) * 3 / 4 / 1024
            logging.debug(f"Final image payload: {final_size_kb:.1f} KB")
            
        return f"data:image/jpeg;base64,{image_base64}"
    except Exception as e:
        logging.error(f"Could not read/process '{image_file}': {e}")
        return None

def load_webcam_urls():
    """Load all webcam URLs"""
    if not WEBCAM_URLS_FILE.exists():
        logging.error(f"File not found: {WEBCAM_URLS_FILE}")
        return []
    
    with open(WEBCAM_URLS_FILE, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    return urls


def verify_with_gpt4o(image_path):
    """Verify yellow car detection with GPT-4o before posting to Bluesky"""
    token = os.getenv("KEY_GITHUB_TOKEN")
    
    if not token:
        logging.warning("GitHub token not available - skipping GPT-4o verification")
        return None
    
    try:
        image_data_url = get_image_data_url(image_path, "jpg", max_size=(800, 600), quality=85)
        if not image_data_url:
            logging.error("Could not prepare image for GPT-4o verification")
            return None
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Does this image show a YELLOW CAR or YELLOW VEHICLE? Answer with only 'yes' or 'no'. Be strict - reject road markings, yellow signs, or anything that isn't an actual yellow car/vehicle."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url,
                                "detail": "low"
                            }
                        }
                    ]
                }
            ],
            "model": "gpt-4o",
            "max_tokens": 10
        }
        
        endpoint = "https://models.inference.ai.azure.com"
        resp = requests.post(f"{endpoint}/chat/completions", json=body, headers=headers, timeout=30)
        
        if resp.status_code != 200:
            logging.warning(f"GPT-4o verification failed with status {resp.status_code}")
            # If GPT-4o fails, log but don't block posting (YOLO26 already checked)
            return None
        
        data = resp.json()
        result = data["choices"][0]["message"]["content"].strip().lower()
        logging.info(f"✅ GPT-4o verification response: {result}")
        
        return "yes" in result
        
    except Exception as e:
        logging.warning(f"Error verifying with GPT-4o: {e}")
        return None


def post_to_bluesky(image_path, alt_text):
    """Post image to Bluesky with alt text"""
    if not BSKY_HANDLE or not BSKY_PASSWORD:
        logging.error("Bluesky credentials not defined")
        return False

    try:
        client = Client()
        client.login(BSKY_HANDLE.strip(), BSKY_PASSWORD.strip())

        image_data_url = get_image_data_url(image_path, "jpg", max_size=(400, 300), quality=70)
        if not image_data_url:
            return False

        header, encoded = image_data_url.split(',', 1)
        image_bytes = base64.b64decode(encoded)
        blob = client.upload_blob(image_bytes).blob

        client.app.bsky.feed.post.create(
            repo=client.me.did,
            record=models.AppBskyFeedPost.Record(
                text="GUL BIL!",
                created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                embed=models.AppBskyEmbedImages.Main(
                    images=[
                        models.AppBskyEmbedImages.Image(
                            image=blob,
                            alt=alt_text
                        )
                    ]
                )
            )
        )
        logging.info("Successfully posted yellow car to Bluesky!")
        return True
    except Exception as e:
        logging.error(f"Error posting to Bluesky: {e}")
        return False

def gulbilbot():
    """Download and test all available webcam images for yellow car detection"""
    
    print("\n" + "=" * 80)
    print("YELLOW CAR DETECTION - FULL SCAN")
    print("=" * 80 + "\n")
    
    # Load URLs
    urls = load_webcam_urls()
    if not urls:
        logging.error("No webcam URLs found")
        return
    
    logging.info(f"Loaded {len(urls)} webcam URLs")
    logging.info("Processing all available images...")
    
    # Process all URLs
    test_urls = urls
    logging.info(f"Selected {len(test_urls)} URLs for testing\n")
    
    # Load YOLO26 model
    logging.info("Loading YOLO26 model...")
    load_yolo_model()
    logging.info("✅ YOLO26 model loaded\n")
    
    # Test images
    results = []
    downloaded = 0
    detected = 0
    posted = 0
    errors = 0
    
    print("=" * 80)
    print("DOWNLOADING AND TESTING IMAGES")
    print("=" * 80 + "\n")
    
    for idx, url in enumerate(test_urls, 1):
        image_name = f"test_{idx:03d}.jpg"
        image_path = TEST_IMAGES_FOLDER / image_name
        
        # Extract webcam ID from URL for logging
        webcam_id = url.split('/images/')[-1] if '/images/' in url else url
        
        # Download image
        total = len(test_urls)
        print(f"[{idx:3d}/{total}] ", end="", flush=True)
        
        if not download_image(url, image_path):
            logging.info(f"Download failed for webcam ID: {webcam_id}")
            print("❌ Download failed")
            errors += 1
            continue
        
        print("✓ Downloaded ", end="", flush=True)
        downloaded += 1
        
        # Test detection
        try:
            result = detect_yellow_car(image_path)
            is_yellow = result["detected"]
            num_boxes = len(result["boxes"]) if is_yellow else 0
            
            if is_yellow:
                detected += 1
                vehicle_types = ", ".join(set(box[4] for box in result["boxes"]))
                print(f"→ YELLOW {vehicle_types.upper()} FOUND ({num_boxes} vehicle(s))")
                
                # Verify with GPT-4o before posting
                logging.info("🔍 Verifying with GPT-4o before posting...")
                gpt4o_confirmed = verify_with_gpt4o(image_path)
                
                if gpt4o_confirmed is False:
                    logging.warning("❌ GPT-4o rejected detection - NOT posting (likely false positive)")
                    print("→ REJECTED by GPT-4o (false positive)")
                elif gpt4o_confirmed is True:
                    logging.info("✅ GPT-4o CONFIRMED yellow car - proceeding to post")
                    # Draw bounding boxes and post to Bluesky
                    annotated_path = draw_bounding_boxes(image_path, result["boxes"])
                    image_to_post = annotated_path if annotated_path else image_path
                    
                    logging.info("📤 Posting to Bluesky...")
                    if post_to_bluesky(image_to_post, alt_text="Yellow car spotted on traffic camera! 🚕"):
                        posted += 1
                        logging.info("✅ Posted successfully!")
                    
                    # Clean up annotated image if it was created
                    if annotated_path and annotated_path != image_path:
                        try:
                            annotated_path.unlink()
                        except Exception:
                            pass
                else:
                    logging.warning("⚠️  GPT-4o verification failed - skipping post to be safe")
                    print("→ GPT-4o verification failed")
            else:
                # Get YOLO detections for debug info
                img = cv2.imread(str(image_path))
                if img is not None:
                    yolo_results = yolo_model(img, verbose=False)
                    vehicle_types = []
                    for res in yolo_results:
                        for det in res.boxes.data.tolist():
                            x1, y1, x2, y2, conf, cls_id = det
                            class_name = yolo_model.names[int(cls_id)]
                            if class_name in ["car", "truck", "bus", "van", "threewheel"]:
                                vehicle_types.append(f"{class_name}({conf:.2f})")
                    
                    if vehicle_types:
                        print(f"→ No yellow car (found: {', '.join(vehicle_types)})")
                    else:
                        print("→ No yellow car (no vehicles detected)")
                else:
                    print("→ No yellow car")
            
            results.append({
                'image': image_name,
                'url': url,
                'webcam_id': webcam_id,
                'detected': is_yellow,
                'boxes': num_boxes
            })
            
        except Exception as e:
            print(f"❌ Detection error: {e}")
            errors += 1
        
        # Clean up downloaded image
        try:
            image_path.unlink()
        except:
            pass
    
    # Summary
    print("\n" + "=" * 80)
    print(f"TEST SUMMARY - {downloaded} IMAGES PROCESSED")
    print("=" * 80)
    print(f"Total images tested: {downloaded}")
    print(f"Yellow cars detected: {detected} ({100*detected/max(1, downloaded):.1f}%)")
    print(f"Posts to Bluesky: {posted}")
    print(f"Download errors: {errors}")
    print("=" * 80 + "\n")
    
    # Statistics
    if detected > 0:
        avg_vehicles = sum(r['boxes'] for r in results if r['detected']) / detected
        print(f"Average vehicles per detection: {avg_vehicles:.1f}")
        print(f"Most common vehicle types:")
        
        vehicle_counts = {}
        for r in results:
            if r['detected']:
                # Would need to track this separately
                pass
    
    # Save detailed results
    results_file = Path("test_results.txt")
    with open(results_file, 'w') as f:
        f.write("Yellow Car Detection - Full Scan Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Downloaded: {downloaded}\n")
        f.write(f"Yellow Cars Found: {detected}\n")
        f.write(f"Detection Rate: {100*detected/max(1, downloaded):.1f}%\n")
        f.write(f"Posts to Bluesky: {posted}\n")
        f.write(f"Errors: {errors}\n\n")
        
        f.write("Detailed Results:\n")
        f.write("-" * 80 + "\n")
        for r in results:
            status = "✓ YES" if r['detected'] else "✗ NO"
            f.write(f"{r['image']:<20} {status:<10} ({r['boxes']} vehicles) - ID: {r['webcam_id']}\n")
    
    logging.info(f"Results saved to: {results_file}\n")
    
    # Cleanup
    try:
        import shutil
        shutil.rmtree(TEST_IMAGES_FOLDER)
        logging.info(f"Cleaned up test folder: {TEST_IMAGES_FOLDER}")
    except:
        pass

if __name__ == "__main__":
    gulbilbot()
