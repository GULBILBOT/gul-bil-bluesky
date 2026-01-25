#!/usr/bin/env python3
"""
Yellow Car Detection Bot with YOLO26
Uses YOLO26 object detection model for detecting yellow cars directly.
"""

import base64
import io
import json
import logging
import os
import random
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import cv2
import numpy as np
from PIL import Image
from atproto import Client, models
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

# Get the project root directory (parent of src folder)
PROJECT_ROOT = Path(__file__).parent.parent
TODAY_FOLDER = PROJECT_ROOT / "today"
TODAY_FOLDER.mkdir(exist_ok=True)
WEBCAM_URLS_FILE = PROJECT_ROOT / "valid_webcam_ids.txt"
SHUFFLE_STATE_FILE = PROJECT_ROOT / "shuffle_state.json"
BSKY_HANDLE = os.getenv("BSKY_HANDLE")
BSKY_PASSWORD = os.getenv("BSKY_PASSWORD")

MAX_RUNTIME_MINUTES = 20
IMAGES_PER_SESSION = 100

# YOLO26 configuration
YOLO_MODEL_PATH = "yolo26n.pt"  # Using YOLO26 Nano for speed
CONF_THRESHOLD = 0.3  # Minimum confidence for vehicle detection
YELLOW_RATIO_THRESHOLD = 0.15  # Minimum yellow proportion in bounding box

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global YOLO model (load once)
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
    """
    global yolo_model
    
    if yolo_model is None:
        if not load_yolo_model():
            return {"detected": False, "boxes": []}
    
    try:
        # Load image with OpenCV
        img = cv2.imread(str(image_path))
        if img is None:
            logging.error(f"Could not load image: {image_path}")
            return {"detected": False, "boxes": []}

        # Run YOLO26 inference
        results = yolo_model(img, verbose=False)
        yellow_boxes = []

        # Parse detections
        for res in results:
            for det in res.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls_id = det

                # Skip low-confidence detections
                if conf < CONF_THRESHOLD:
                    continue

                # Check if detected class is a yellow vehicle
                # Accept: car, bus, truck, van, threewheel
                # Reject: motorcycle
                class_name = yolo_model.names[int(cls_id)]
                if class_name not in ["car", "truck", "bus", "van", "threewheel"]:
                    continue

                # Crop the bounding box region
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
                
                # HSV range for yellow (adjusted for traffic cameras)
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


def draw_bounding_boxes(image_path, boxes, output_path=None):
    """
    Draw bounding boxes on detected yellow vehicles.
    Args:
        image_path: Path to original image
        boxes: List of (x1, y1, x2, y2, class_name, conf, yellow_ratio) tuples
        output_path: Path to save annotated image (if None, returns modified image)
    Returns:
        Path to annotated image or None if failed
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Draw bounding boxes
        for x1, y1, x2, y2, class_name, conf, yellow_ratio in boxes:
            # Draw yellow rectangle (bright yellow in BGR: 0, 255, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
            
            # Draw label with class name and confidence
            label = f"{class_name} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Get text size to draw background
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width + 5, y1), (0, 255, 255), -1)
            
            # Draw text
            cv2.putText(img, label, (x1 + 2, y1 - 5), font, font_scale, (0, 0, 0), thickness)
        
        # Save or return the annotated image
        if output_path is None:
            output_path = image_path.parent / f"annotated_{image_path.name}"
        
        cv2.imwrite(str(output_path), img)
        logging.debug(f"Saved annotated image to {output_path}")
        return output_path
        
    except Exception as e:
        logging.error(f"Error drawing bounding boxes: {e}")
        return None


def load_shuffle_state():
    if SHUFFLE_STATE_FILE.exists():
        try:
            with open(SHUFFLE_STATE_FILE, 'r') as f:
                state = json.load(f)
                if not isinstance(state.get("shuffled_urls"), list):
                    return {"shuffled_urls": [], "current_index": 0, "stats": {"total_processed": 0, "total_posted": 0}}
                return state
        except Exception as e:
            logging.warning(f"Could not load shuffle state: {e}")
    return {"shuffled_urls": [], "current_index": 0, "stats": {"total_processed": 0, "total_posted": 0}}


def save_shuffle_state(state):
    try:
        with open(SHUFFLE_STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        logging.error(f"Could not save shuffle state: {e}")


def get_shuffled_urls():
    if not WEBCAM_URLS_FILE.exists():
        logging.error(f"Webcam URLs file not found: {WEBCAM_URLS_FILE}")
        return [], 0, {}

    with open(WEBCAM_URLS_FILE, "r") as f:
        all_urls = [line.strip() for line in f if line.strip()]

    state = load_shuffle_state()

    if not state["shuffled_urls"] or state["current_index"] >= len(state["shuffled_urls"]):
        logging.info(f"Shuffling {len(all_urls)} webcam URLs for fair processing")
        state["shuffled_urls"] = all_urls.copy()
        random.shuffle(state["shuffled_urls"])
        state["current_index"] = 0
        cycle_num = state.get("cycle_count", 0) + 1
        state["cycle_count"] = cycle_num
        save_shuffle_state(state)

    return state["shuffled_urls"], state["current_index"], state.get("stats", {"total_processed": 0, "total_posted": 0})


def update_shuffle_state(new_index, stats_update=None):
    state = load_shuffle_state()
    state["current_index"] = new_index
    if stats_update:
        if "stats" not in state:
            state["stats"] = {"total_processed": 0, "total_posted": 0}
        for key, value in stats_update.items():
            state["stats"][key] = state["stats"].get(key, 0) + value
    save_shuffle_state(state)


def download_image(url, dest, timeout=10):
    try:
        resp = requests.get(url, allow_redirects=True, timeout=timeout, stream=True)
        if resp.status_code != 200:
            logging.debug(f"Failed to download {url}: Status {resp.status_code}")
            return False
        
        # Verify we have content before writing
        if not resp.content or len(resp.content) == 0:
            logging.debug(f"Empty response from {url}")
            return False
        
        # Check Content-Length header if available
        content_length = resp.headers.get('content-length')
        if content_length and int(content_length) == 0:
            logging.debug(f"Content-Length is 0 for {url}")
            return False
        
        # Ensure parent directory exists
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        with open(dest, "wb") as f:
            f.write(resp.content)
        
        # Verify file was written
        if not dest.exists():
            logging.debug(f"File not created: {dest}")
            return False
        
        file_size = dest.stat().st_size
        if file_size == 0:
            logging.debug(f"File is empty after write: {dest}")
            dest.unlink()
            return False
        
        return True
        
    except Exception as e:
        logging.debug(f"Exception downloading {url}: {e}")
        # Clean up partial file if it exists
        if dest.exists():
            try:
                dest.unlink()
            except:
                pass
        return False





def get_image_data_url(image_file, image_format, max_size=(800, 600), quality=85):
    """
    Get base64 data URL for image, with automatic resizing to avoid 413 errors
    """
    try:
        # Open and potentially resize image
        with Image.open(image_file) as img:
            img = img.convert('RGB')
            
            # Check if image needs resizing
            original_size = img.size
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                # Resize maintaining aspect ratio
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                logging.debug(f"Resized image from {original_size} to {img.size}")
            
            # Save to bytes with compression
            import io
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG', quality=quality, optimize=True)
            img_bytes.seek(0)
            
            # Convert to base64
            image_base64 = base64.b64encode(img_bytes.getvalue()).decode()
            
            # Log final size
            final_size_kb = len(image_base64) * 3 / 4 / 1024  # Approximate KB
            logging.debug(f"Final image payload: {final_size_kb:.1f} KB")
            
        return f"data:image/jpeg;base64,{image_base64}"
    except Exception as e:
        logging.error(f"Could not read/process '{image_file}': {e}")
        return None


def post_to_bluesky(image_path, alt_text):
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


def main():
    """Main bot loop - download images and detect yellow cars"""
    start_time = datetime.now()
    max_end_time = start_time + timedelta(minutes=MAX_RUNTIME_MINUTES)

    logging.info(f"Starting Yellow Car Bot with YOLO26")
    logging.info(f"Max runtime: {MAX_RUNTIME_MINUTES} minutes")
    logging.info(f"Images per session: {IMAGES_PER_SESSION}")

    # Pre-load YOLO26 model
    if not load_yolo_model():
        logging.error("Failed to load YOLO26 model - exiting")
        return

    urls, current_index, current_stats = get_shuffled_urls()
    if not urls:
        logging.error("No URLs available")
        return

    logging.info(f"Loaded {len(urls)} webcam URLs")
    logging.info(f"Resuming from position {current_index}/{len(urls)}")
    logging.info(f"All-time stats: {current_stats.get('total_processed', 0)} processed, {current_stats.get('total_posted', 0)} posted")

    session_processed = 0
    session_yellow_found = 0
    session_posted = 0
    final_index = current_index

    try:
        # Use the same straightforward progress style as test_100_images.py
        for i in range(current_index, min(current_index + IMAGES_PER_SESSION, len(urls))):
            if datetime.now() >= max_end_time:
                logging.info("Time limit reached, stopping gracefully")
                break

            url = urls[i]
            timestamp = int(time.time())
            image_name = f"cam_{i + 1}_{timestamp}.jpg"
            image_path = TODAY_FOLDER / image_name

            # Download image (mirror the simple stdout progress from the test script)
            print(f"[{i+1:3d}/{len(urls)}] ", end="", flush=True)
            if not download_image(url, image_path):
                print("❌ Download failed")
                continue

            print("✓ Downloaded ", end="", flush=True)
            session_processed += 1

            # Run YOLO26 detection
            try:
                detection_result = detect_yellow_car(image_path)
            except Exception as e:
                print(f"❌ Detection error: {e}")
                continue

            if detection_result["detected"]:
                session_yellow_found += 1
                vehicle_types = ", ".join(sorted(set(box[4] for box in detection_result["boxes"]))) or "vehicle"
                num_boxes = len(detection_result["boxes"])
                print(f"→ YELLOW {vehicle_types.upper()} FOUND ({num_boxes} vehicle(s))")

                # Draw bounding boxes on the image
                annotated_path = draw_bounding_boxes(image_path, detection_result["boxes"])
                image_to_post = annotated_path if annotated_path else image_path

                logging.info("📤 Posting to Bluesky...")
                if post_to_bluesky(image_to_post, alt_text="Yellow car spotted on traffic camera! 🚕"):
                    session_posted += 1
                    logging.info("✅ Posted successfully!")

                # Clean up annotated image if it was created
                if annotated_path and annotated_path != image_path:
                    try:
                        annotated_path.unlink()
                    except Exception:
                        pass
            else:
                # Mirror the debug output style from test_100_images.py
                try:
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
                except Exception as e:
                    print(f"→ No yellow car (debug error: {e})")

            # Clean up downloaded image
            try:
                image_path.unlink()
            except Exception:
                pass

            time.sleep(0.5)

        final_index = min(current_index + session_processed, len(urls))
        stats_update = {
            "total_processed": session_processed,
            "total_posted": session_posted
        }
        update_shuffle_state(final_index, stats_update)

    except KeyboardInterrupt:
        logging.info("\n⏸️  Interrupted, saving progress...")
        final_index = current_index + session_processed
        stats_update = {"total_processed": session_processed, "total_posted": session_posted}
        update_shuffle_state(final_index, stats_update)
    except Exception as e:
        logging.error(f"Unexpected error in main loop: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    runtime = datetime.now() - start_time
    updated_stats = load_shuffle_state().get("stats", {})

    logging.info("\n" + "=" * 80)
    logging.info("SESSION SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Runtime: {runtime.total_seconds():.1f} seconds")
    logging.info(f"Images processed: {session_processed}")
    logging.info(f"Yellow cars detected: {session_yellow_found}")
    logging.info(f"Posts to Bluesky: {session_posted}")
    logging.info(f"Progress: {final_index}/{len(urls)}")
    logging.info(f"All-time totals: {updated_stats.get('total_processed', 0)} processed, {updated_stats.get('total_posted', 0)} posted")
    logging.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()