import os
import json
import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = r"D:\RESUME\Origin\Task\Drywall-Join-Detect.v2i.coco"
SPLITS = ["train", "valid"]

# -----------------------------
# PROCESS FUNCTION
# -----------------------------
def process_split(split):

    print(f"\nProcessing {split}...")

    json_path = os.path.join(BASE_DIR,f"_annotations_{split}.coco.json")
    images_dir = os.path.join(BASE_DIR, split)
    masks_dir = os.path.join(BASE_DIR, f"{split}_masks")

    os.makedirs(masks_dir, exist_ok=True)

    # load COCO
    with open(json_path, "r") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]

    # group annotations
    ann_per_image = defaultdict(list)
    for ann in annotations:
        ann_per_image[ann["image_id"]].append(ann)

    # process images
    for img in tqdm(images):
        img_id = img["id"]
        filename = img["file_name"]
        height = img["height"]
        width = img["width"]

        mask = np.zeros((height, width), dtype=np.uint8)

        anns = ann_per_image.get(img_id, [])

        for ann in anns:
            bbox = ann.get("bbox", None)

            if bbox is None or len(bbox) != 4:
                continue

            # COCO format: [x, y, w, h]
            x, y, w, h = bbox

            # compute corners (float → int carefully)
            x1 = int(np.floor(x))
            y1 = int(np.floor(y))
            x2 = int(np.ceil(x + w))
            y2 = int(np.ceil(y + h))

            # ✅ clip BEFORE slicing
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            # ✅ skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            # fill rectangle
            mask[y1:y2, x1:x2] = 255

        # save mask
        mask_name = os.path.splitext(filename)[0] + ".png"
        cv2.imwrite(os.path.join(masks_dir, mask_name), mask)

    print(f"{split} done. Masks saved in {masks_dir}")


# -----------------------------
# RUN
# -----------------------------
for split in SPLITS:
    process_split(split)

print("\nAll done!")