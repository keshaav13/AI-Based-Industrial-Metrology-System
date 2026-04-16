import cv2
import numpy as np
import os
from ultralytics import YOLO
from collections import deque

# ==========================================================
# CONFIGURATION
# ==========================================================
weights_path     = r"C:\Users\PMTC ELE\Desktop\ins\train8\weights\best.pt"
test_images_path = r"C:\Users\PMTC ELE\Desktop\ins\test_Images"
output_path      = r"C:\Users\PMTC ELE\Desktop\ins\outputs"

# Real known diameter
REAL_DIAMETER_MM = 69.90

# Calibration diameter in pixels (best measured reference image)
CALIBRATION_PX = 236

# mm per pixel
mm_per_px = REAL_DIAMETER_MM / CALIBRATION_PX

# Allowed stable range (prevent enlarge / shrink issue)
MIN_VALID_PX = 228
MAX_VALID_PX = 240

# Moving average window
SMOOTH_WINDOW = 10

# ==========================================================
# LOAD MODEL
# ==========================================================
model = YOLO(weights_path)
os.makedirs(output_path, exist_ok=True)

# Stores previous diameters
diameter_history = deque(maxlen=SMOOTH_WINDOW)

# ==========================================================
# EDGE SCORE
# ==========================================================
def score_circle(edges, cx, cy, r):
    h, w = edges.shape
    hit = 0
    total = 0

    for ang in range(0, 360, 4):

        # Ignore top blocked hand region
        if 65 <= ang <= 115:
            continue

        rad = np.deg2rad(ang)

        px = int(cx + r * np.cos(rad))
        py = int(cy + r * np.sin(rad))

        if 0 <= px < w and 0 <= py < h:
            total += 1
            if edges[py, px] > 0:
                hit += 1

    if total == 0:
        return 0

    return hit / total


# ==========================================================
# DETECT OUTER CIRCLE
# ==========================================================
def detect_outer_circle(roi):

    h, w = roi.shape[:2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Contrast enhance
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (7,7), 1.8)

    edges = cv2.Canny(blur, 50, 160)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=int(min(h,w)*0.4),
        param1=120,
        param2=25,
        minRadius=int(min(h,w)*0.42),
        maxRadius=int(min(h,w)*0.56)
    )

    if circles is None:
        return None, None, None, None

    circles = np.round(circles[0]).astype(int)

    roi_cx = w / 2
    roi_cy = h / 2
    max_dist = np.sqrt(roi_cx**2 + roi_cy**2)

    best_circle = None
    best_score = -1

    for c in circles:

        cx, cy, r = c

        edge_score = score_circle(edges, cx, cy, r)

        dist = np.sqrt((cx-roi_cx)**2 + (cy-roi_cy)**2)
        center_score = 1 - (dist / max_dist)

        radius_score = r / (min(h,w)*0.56)

        total_score = (
            edge_score * 0.55 +
            center_score * 0.20 +
            radius_score * 0.25
        )

        if total_score > best_score:
            best_score = total_score
            best_circle = c

    if best_circle is None:
        return None, None, None, None

    cx, cy, r = best_circle
    diameter_px = 2 * r

    return cx, cy, r, diameter_px


# ==========================================================
# MAIN LOOP
# ==========================================================
all_results = []

for file_name in os.listdir(test_images_path):

    img_path = os.path.join(test_images_path, file_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    results = model(img)

    for result in results:

        boxes = result.boxes.xyxy.cpu().numpy()

        for box in boxes:

            x1, y1, x2, y2 = map(int, box)

            roi = img[y1:y2, x1:x2]

            cx, cy, r, dia_px = detect_outer_circle(roi)

            if dia_px is None:
                print(file_name, "Detection failed")
                continue

            # =====================================
            # Reject abnormal enlarge/shrink values
            # =====================================
            if MIN_VALID_PX <= dia_px <= MAX_VALID_PX:
                diameter_history.append(dia_px)

            # If no values yet
            if len(diameter_history) == 0:
                stable_px = dia_px
            else:
                stable_px = np.mean(diameter_history)

            stable_mm = stable_px * mm_per_px

            all_results.append(stable_px)

            # Full image coords
            cx_full = x1 + cx
            cy_full = y1 + cy
            r_draw = int(stable_px / 2)

            # Draw bounding box
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

            # Draw stable circle
            cv2.circle(img, (cx_full, cy_full), r_draw, (0,255,0), 3)

            # Center point
            cv2.circle(img, (cx_full, cy_full), 4, (0,0,255), -1)

            # Diameter line
            cv2.line(
                img,
                (cx_full-r_draw, cy_full),
                (cx_full+r_draw, cy_full),
                (255,255,0),
                2
            )

            # Text
            cv2.putText(
                img,
                f"{stable_mm:.2f} mm",
                (x1, y1-30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,0,255),
                2
            )

            cv2.putText(
                img,
                f"{stable_px:.0f}px",
                (x1, y1-8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

            print(f"{file_name} --> Stable: {stable_px:.2f}px | {stable_mm:.2f} mm")

    save_path = os.path.join(output_path, file_name)
    cv2.imwrite(save_path, img)

# ==========================================================
# FINAL REPORT
# ==========================================================
if len(all_results) > 0:

    avg_px = np.mean(all_results)
    avg_mm = avg_px * mm_per_px

    print("\n================================")
    print(f"Final Stable Average : {avg_px:.2f}px")
    print(f"Final Stable Average : {avg_mm:.2f} mm")
    print("================================")

print("Done. Outputs saved in:", output_path)