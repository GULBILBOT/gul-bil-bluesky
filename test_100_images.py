#!/usr/bin/env python3
"""
Download and test 100 random webcam images for yellow car detection
"""

import base64
import os
import random
import requests
import logging
from pathlib import Path
from datetime import datetime, timezone
from atproto import Client, models
from dotenv import load_dotenv
from src.main import load_yolo_model, detect_yellow_car, download_image, draw_bounding_boxes, get_image_data_url

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

TEST_FOLDER = Path("test_100_images")
TEST_FOLDER.mkdir(exist_ok=True)

# Read webcam URLs
WEBCAM_URLS_FILE = Path("valid_webcam_ids.txt")

# Bluesky credentials
BSKY_HANDLE = os.getenv("BSKY_HANDLE")
BSKY_PASSWORD = os.getenv("BSKY_PASSWORD")

def load_webcam_urls():
    """Load all webcam URLs"""
    if not WEBCAM_URLS_FILE.exists():
        logging.error(f"File not found: {WEBCAM_URLS_FILE}")
        return []
    
    with open(WEBCAM_URLS_FILE, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    return urls


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

def test_100_images():
    """Download and test 100 random webcam images"""
    
    print("\n" + "=" * 80)
    print("YELLOW CAR DETECTION - 100 IMAGE TEST")
    print("=" * 80 + "\n")
    
    # Load URLs
    urls = load_webcam_urls()
    if not urls:
        logging.error("No webcam URLs found")
        return
    
    logging.info(f"Loaded {len(urls)} webcam URLs")
    logging.info("Selecting 100 random images...")
    
    # Select 100 random URLs
    test_urls = random.sample(urls, min(100, len(urls)))
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
        image_path = TEST_FOLDER / image_name
        
        # Download image
        print(f"[{idx:3d}/100] ", end="", flush=True)
        
        if not download_image(url, image_path):
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
                # Get YOLO detections for debug info
                import cv2
                img = cv2.imread(str(image_path))
                if img is not None:
                    from src.main import yolo_model
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
    print("TEST SUMMARY - 100 IMAGES")
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
    results_file = Path("test_100_results.txt")
    with open(results_file, 'w') as f:
        f.write("Yellow Car Detection - 100 Image Test Results\n")
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
            f.write(f"{r['image']:<20} {status:<10} ({r['boxes']} vehicles)\n")
    
    logging.info(f"Results saved to: {results_file}\n")
    
    # Cleanup
    try:
        import shutil
        shutil.rmtree(TEST_FOLDER)
        logging.info(f"Cleaned up test folder: {TEST_FOLDER}")
    except:
        pass

if __name__ == "__main__":
    test_100_images()
