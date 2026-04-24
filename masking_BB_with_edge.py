import os
import json
import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict

BASE_DIR = r"D:\RESUME\Origin\Task\Drywall-Join-Detect.v2i.coco"
SPLITS = ["train", "valid"]

def process_split(split):

    print(f"\nProcessing {split}...")

    json_path = os.path.join(BASE_DIR,f"_annotations_{split}.coco.json")
    images_dir = os.path.join(BASE_DIR, split)
    masks_dir = os.path.join(BASE_DIR, f"{split}_masks_2")

    os.makedirs(masks_dir, exist_ok=True)

    with open(json_path, "r") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]

    ann_per_image = defaultdict(list)
    for ann in annotations:
        ann_per_image[ann["image_id"]].append(ann)

    for img in tqdm(images):
        img_id = img["id"]
        filename = img["file_name"]
        height = img["height"]
        width = img["width"]

        image = cv2.imread(os.path.join(images_dir, filename))
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros((height, width), dtype=np.uint8)

        anns = ann_per_image.get(img_id, [])

        for ann in anns:
            bbox = ann.get("bbox", None)
            if bbox is None:
                continue

            x, y, w, h = bbox

            x1 = int(max(0, np.floor(x)))
            y1 = int(max(0, np.floor(y)))
            x2 = int(min(width, np.ceil(x + w)))
            y2 = int(min(height, np.ceil(y + h)))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = gray[y1:y2, x1:x2]

            # -----------------------------
            # EDGE DETECTION
            # -----------------------------
            crop = cv2.GaussianBlur(crop, (5,5), 0)
            edges = cv2.Canny(crop, 50, 150)

            edge_ratio = np.mean(edges > 0)

            # -----------------------------
            # FALLBACK (no edges)
            # -----------------------------
            if edge_ratio < 0.02:
                mask[y1:y2, x1:x2] = 255
                continue

            # -----------------------------
            # MORPHOLOGY (controlled)
            # -----------------------------
            kernel = np.ones((3, 3), np.uint8)

            edges = cv2.dilate(edges, kernel, iterations=1)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

            # -----------------------------
            # THICKEN INTO REGION
            # -----------------------------
            region = cv2.dilate(edges, kernel, iterations=2)

            region = (region > 0).astype(np.uint8) * 255

            # -----------------------------
            # REMOVE SMALL COMPONENTS  ← ADD HERE
            # -----------------------------
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)

            clean = np.zeros_like(region)
            bbox_area = (x2 - x1) * (y2 - y1)
            min_area = max(30, int(0.001 * bbox_area))

            for i in range(1, num_labels):  # skip background
                area = stats[i, cv2.CC_STAT_AREA]
                if area > min_area:
                    clean[labels == i] = 255

            region = clean

            # -----------------------------
            # CHECK REGION SIZE
            # -----------------------------

            region_ratio = np.mean(region > 0)

            # -----------------------------
            # FALLBACK (too big = noise)
            # -----------------------------
            # too small → fallback
            if region_ratio < 0.01:
                mask[y1:y2, x1:x2] = 255
                continue

            if region_ratio > 0.75:
                mask[y1:y2, x1:x2] = 255
            else:
                mask[y1:y2, x1:x2] = region

        mask_name = os.path.splitext(filename)[0] + ".png"
        cv2.imwrite(os.path.join(masks_dir, mask_name), mask)

    print(f"{split} done!")

for split in SPLITS:
    process_split(split)

print("All done!")