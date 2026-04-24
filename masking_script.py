import os
import json
import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict

# -----------------------------
# PATHS (EDIT THESE)
# -----------------------------
COCO_JSON = r"D:\RESUME\Origin\Task\_annotations.coco.json"
IMAGES_DIR = r"D:\RESUME\Origin\Task\cracks.coco\train"
MASKS_DIR = r"D:\RESUME\Origin\Task\cracks.coco\masked"

os.makedirs(MASKS_DIR, exist_ok=True)

# -----------------------------
# LOAD COCO
# -----------------------------
with open(COCO_JSON, "r") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]

# -----------------------------
# GROUP ANNOTATIONS
# -----------------------------
ann_per_image = defaultdict(list)
for ann in annotations:
    ann_per_image[ann["image_id"]].append(ann)

# -----------------------------
# CREATE MASKS
# -----------------------------
print("Generating masks...")

for img in tqdm(images):
    img_id = img["id"]
    filename = img["file_name"]
    height = img["height"]
    width = img["width"]

    mask = np.zeros((height, width), dtype=np.uint8)
    anns = ann_per_image.get(img_id, [])

    for ann in anns:
        segmentations = ann.get("segmentation", [])
        if not segmentations:
            continue

        for seg in segmentations:
            if len(seg) < 6 or len(seg) % 2 != 0:
                continue

            pts = np.array(seg).reshape(-1, 2).astype(np.float32)
            pts = np.round(pts).astype(np.int32)

            pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)

            cv2.fillPoly(mask, [pts], 255)

    mask_name = os.path.splitext(filename)[0] + ".png"
    cv2.imwrite(os.path.join(MASKS_DIR, mask_name), mask)

print("Masks created!")