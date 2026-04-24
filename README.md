# CLIPSeg Crack & Drywall Segmentation

Prompt-based segmentation of wall cracks and drywall seams using fine-tuned CLIPSeg.

## 🚀 Overview

This project fine-tunes a vision-language model to generate binary segmentation masks conditioned on prompts:

* `segment crack`
* `segment drywall seam`

The model learns both tasks within a single architecture using prompt conditioning.

## 🧠 Model

* Model: CLIPSeg (`CIDAS/clipseg-rd64-refined`)
* Framework: PyTorch
* Input Resolution: ~352×352 (internal)
* Output: Binary masks resized back to original size (640×640)

## 📊 Dataset

* Crack images: ~5369
* Drywall images: ~1022

Train/Validation split used.

## ⚙️ Training

### Loss

* BCE + Dice Loss (λ = 0.6)

### Optimizer

* AdamW (lr = 1e-5)

### Scheduler

* ReduceLROnPlateau (monitor: mIoU)

### Key techniques

* Weighted sampling (handles class imbalance)
* Prompt-based conditioning
* Gradient clipping

## 📈 Results

| Metric | Value |
|--------|-------|
| IoU (Crack) | 0.439 |
| IoU (Drywall) | 0.596 |
| mIoU | 0.518 |
| mDice | 0.662 |

## 📥 Model Predictions

Access predictions from both models (Experiment 1 & Experiment 2):

* [Predictions on Google Drive](https://drive.google.com/drive/folders/12Xkuo2P0K_5BKM_XlBv2beWipf23J6CP?usp=sharing)

The drive contains:
* Experiment 1 predictions (baseline model)
* Experiment 2 predictions (improved model)
* Both crack and drywall segmentation outputs

  Access predictions from both models (Experiment 1 & Experiment 2):

Predictions on Google Drive

Predictions:

* Model 1 (ZIP): https://drive.google.com/file/d/1_id87K7cETFAeoBqay9zSDCRnopAMsu_/view?usp=sharing
* Model 2 (ZIP): https://drive.google.com/file/d/13AcEXmAzhP9PtwMgskvFctTO30MjcK9S/view?usp=drive_link



## 📥 Model Weights
Pretrained weights are available here:

* Model 2 (best): https://www.kaggle.com/datasets/kartikjoshi24/model-2-clipseg-crack-and-drywall-segmentation
* Model 1 (baseline): https://www.kaggle.com/datasets/kartikjoshi24/model-1-clipseg-crack-and-drywall-segmentation




## ⚡ Performance

* Inference: ~0.042 sec/image (~23 FPS)
* Model size: ~603 MB
* Training time: ~2 hours (T4 GPU)

## 📦 Output Format

* PNG (single-channel)
* Values: `{0, 255}`
* Same resolution as input
* Filename format:

```
<image_id>__<prompt>.png
```

## ⚠️ Limitations

* Very thin cracks may be partially missed
* Low-contrast drywall regions are challenging
* Ground truth masks for drywall are sometimes coarse


## 🏁 Summary

Combining BCE + Dice loss and balanced sampling significantly improves segmentation quality, especially for the underrepresented drywall class, achieving strong and consistent performance across both tasks.
