import json
from collections import defaultdict

# path to your COCO file
coco_path = r"D:\RESUME\Origin\Task\_annotations.coco.json"  # adjust if needed

with open(coco_path, "r") as f:
    data = json.load(f)

# ----------------------------
# 1. CHECK CATEGORIES
# ----------------------------
categories = data.get("categories", [])
print("\n📌 Categories:")
for cat in categories:
    print(cat)

print(f"\nTotal categories: {len(categories)}")

# ----------------------------
# 2. COUNT IMAGES
# ----------------------------
images = data.get("images", [])
print(f"\n📌 Total images: {len(images)}")

# ----------------------------
# 3. COUNT ANNOTATIONS
# ----------------------------
annotations = data.get("annotations", [])
print(f"\n📌 Total annotations: {len(annotations)}")

# ----------------------------
# 4. IMAGES WITH ANNOTATIONS
# ----------------------------
image_ids_with_annotations = set([ann["image_id"] for ann in annotations])
print(f"\n📌 Images that have annotations: {len(image_ids_with_annotations)}")

# ----------------------------
# 5. MULTIPLE ANNOTATIONS PER IMAGE
# ----------------------------
ann_per_image = defaultdict(int)

for ann in annotations:
    ann_per_image[ann["image_id"]] += 1

multi_ann_images = [img_id for img_id, count in ann_per_image.items() if count > 1]

print(f"\n📌 Images with multiple annotations: {len(multi_ann_images)}")

# show few examples
print("Examples (image_id : count):")
for img_id in multi_ann_images[:5]:
    print(img_id, ":", ann_per_image[img_id])

# ----------------------------
# 6. MULTIPLE SEGMENTATIONS PER ANNOTATION
# ----------------------------
multi_seg = 0

for ann in annotations:
    if len(ann.get("segmentation", [])) > 1:
        multi_seg += 1

print(f"\n📌 Annotations with multiple segmentations: {multi_seg}")

# ----------------------------
# 7. CATEGORY DISTRIBUTION
# ----------------------------
cat_count = defaultdict(int)

for ann in annotations:
    cat_count[ann["category_id"]] += 1

print("\n📌 Category distribution:")
for k, v in cat_count.items():
    print(f"Category {k}: {v} annotations")