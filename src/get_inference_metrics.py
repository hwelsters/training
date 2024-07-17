import os

import numpy as np
import pandas as pd

from lora_inference import LoraSamInference
from segment_anything_utils import show_black_and_white_image, save_black_and_white_image, SegmentAnythingSession
from segmentation_metrics import segmentation_classification_report, segmentation_dice_score, segmentation_jaccard_score

NUM_INDEXES = 10

for model_path in [
   "normal",
   "dust-0.4",
   "fog-0.4",
   "mapleleaf-0.4"
]:
  rank = 8
  lora_sam = LoraSamInference("./model_checkpoint/sam_vit_b_01ec64.pth", f"finetuned_weights/{model_path}/lora_rank{rank}.safetensors", rank)
  performance = []
  for image_path in [
    "normal",
    "dust-0.4", 
    "fog-0.4", 
    "mapleleaf-0.4", 
  ]:
    
    count = 0
    average_accuracy = 0
    average_precision_foreground = 0
    average_recall_foreground = 0
    average_f1_foreground = 0
    average_precision_background = 0
    average_recall_background = 0
    average_f1_background = 0
    average_dice_score = 0
    average_jaccard_score = 0
    for sample_index in range(1, 10):
      # This is done because the last two samples were used in the training set.
      # if sample_index > 7: continue

      for index in range(0, NUM_INDEXES):
        # ==============================
        # Paths
        # ==============================
        full_image_path = f"images/{image_path}/sample_{sample_index}/{index}/Scene.png"
        full_mask_path = f"images/{image_path}/sample_{sample_index}/{index}/Masks.png"

        full_cache_path = f"cache/lora_predictions/rank_{rank}/{model_path}/{image_path}/sample_{sample_index}/{index}/prediction.npy"
        full_logit_cache_path = f"cache/lora_predictions/rank_{rank}/{model_path}/{image_path}/sample_{sample_index}/{index}/logits.npy"
        full_output_path = f"output/lora_predictions/rank_{rank}/model_{model_path}/{image_path}/sample_{sample_index}/{index}/prediction.png"

        # Create directories if they don't exist
        if not os.path.exists(f"output/lora_predictions/rank_{rank}/model_{model_path}/{image_path}/sample_{sample_index}/{index}"):
            os.makedirs(f"output/lora_predictions/rank_{rank}/model_{model_path}/{image_path}/sample_{sample_index}/{index}")

        # ==============================
        # Inference
        # ==============================
        ground_truth = SegmentAnythingSession.filter_colors(full_mask_path, [(255, 255, 255)])

        masks = lora_sam.cached_predict(full_cache_path, full_image_path, full_mask_path, verbose=True)
        logical_or_mask = masks[0]
        for mask in masks:
            logical_or_mask = np.logical_or(logical_or_mask, mask)
        save_black_and_white_image(logical_or_mask, full_output_path)

        logit_masks = lora_sam.cached_predict(full_logit_cache_path, full_image_path, full_mask_path, return_logits=True, verbose=True)
        combined_logit_mask = logit_masks[0]
        for mask in logit_masks:
          combined_logit_mask = np.add(combined_logit_mask, mask)

        # ==============================
        # Metrics
        # ==============================
        classification_report = segmentation_classification_report(logical_or_mask, ground_truth)
        average_accuracy += classification_report["accuracy"]
        average_precision_foreground += classification_report["foreground"]["precision"]
        average_recall_foreground += classification_report["foreground"]["recall"]
        average_f1_foreground += classification_report["foreground"]["f1-score"]
        average_precision_background += classification_report["background"]["precision"]
        average_recall_background += classification_report["background"]["recall"]
        average_f1_background += classification_report["background"]["f1-score"]
        average_dice_score += segmentation_dice_score(logical_or_mask, ground_truth)
        average_jaccard_score += segmentation_jaccard_score(logical_or_mask, ground_truth)
        count += 1

    performance.append({
        "image_path": full_image_path,
        "average_accuracy": average_accuracy / count,
        "average_precision (foreground)": average_precision_foreground / count,
        "average_recall (foreground)": average_recall_foreground / count,
        "average_f1 (foreground)": average_f1_foreground / count,
        "average_precision (background)": average_precision_background / count,
        "average_recall (background)": average_recall_background / count,
        "average_f1 (background)": average_f1_background / count,
        "average_dice_score": average_dice_score / count,
        "average_jaccard_score": average_jaccard_score / count,
    })
    
  df = pd.DataFrame(performance)

  if not os.path.exists(f"output/lora_metrics/rank_{rank}/model_{model_path}"):
      os.makedirs(f"output/lora_metrics/rank_{rank}/model_{model_path}") 
  df.to_csv(f"output/lora_metrics/rank_{rank}/model_{model_path}/performance.csv", index=False)