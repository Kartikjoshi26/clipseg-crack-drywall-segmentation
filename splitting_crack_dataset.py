import os
import shutil
import random
from tqdm import tqdm

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = r"D:\RESUME\Origin\Task\cracks.coco"

IMAGE_DIR = os.path.join(BASE_DIR, "train")    # original images
MASK_DIR  = os.path.join(BASE_DIR, "masked")   # masks

OUTPUT_DIR = r"D:\RESUME\Origin\Task\cracks_split"

# -----------------------------
# CREATE OUTPUT STRUCTURE
# -----------------------------
for split in ["train", "val"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "masks"), exist_ok=True)

# -----------------------------
# GET IMAGE FILES
# -----------------------------
images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

# -----------------------------
# SHUFFLE
# -----------------------------
random.seed(42)
random.shuffle(images)

# -----------------------------
# SPLIT 80/20
# -----------------------------
split_idx = int(0.8 * len(images))
train_imgs = images[:split_idx]
val_imgs   = images[split_idx:]

print(f"Total: {len(images)}")
print(f"Train: {len(train_imgs)}")
print(f"Val: {len(val_imgs)}")

# -----------------------------
# COPY FUNCTION
# -----------------------------
def copy_data(img_list, split):

    for img_name in tqdm(img_list, desc=f"{split}"):

        img_src = os.path.join(IMAGE_DIR, img_name)

        # mask name (convert extension to .png)
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_src = os.path.join(MASK_DIR, mask_name)

        img_dst = os.path.join(OUTPUT_DIR, split, "images", img_name)
        mask_dst = os.path.join(OUTPUT_DIR, split, "masks", mask_name)

        # copy image
        if os.path.exists(img_src):
            shutil.copy2(img_src, img_dst)
        else:
            print(f"Missing image: {img_name}")

        # copy mask
        if os.path.exists(mask_src):
            shutil.copy2(mask_src, mask_dst)
        else:
            print(f"Missing mask: {mask_name}")

# -----------------------------
# RUN
# -----------------------------
copy_data(train_imgs, "train")
copy_data(val_imgs, "val")

print("\n✅ Split completed!")