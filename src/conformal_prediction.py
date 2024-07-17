import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from lora_inference import LoraSamInference
from segment_anything_utils import show_black_and_white_image, save_black_and_white_image, SegmentAnythingSession
from segmentation_metrics import segmentation_classification_report, segmentation_dice_score, segmentation_jaccard_score

NUM_INDEXES = 10
ALPHA = 0.2

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
    # This is done because the last two samples were used in the training set.
    cal_sgmd = []
    cal_gt_masks = []
    val_sgmd = []
    val_gt_masks = []
    for sample_index in range(1, 10):
      for index in range(0, NUM_INDEXES):
        full_image_path = f"images/{image_path}/sample_{sample_index}/{index}/Scene.png"
        full_mask_path = f"images/{image_path}/sample_{sample_index}/{index}/Masks.png"
        full_cache_path = f"cache/lora_predictions/rank_{rank}/{model_path}/{image_path}/sample_{sample_index}/{index}/prediction.npy"
        full_logit_cache_path = f"cache/lora_predictions/rank_{rank}/{model_path}/{image_path}/sample_{sample_index}/{index}/logits.npy"
        ground_truth = SegmentAnythingSession.filter_colors(full_mask_path, [(255, 255, 255)])
        logit_masks = lora_sam.cached_predict(full_logit_cache_path, full_image_path, full_mask_path, return_logits=True)
        combined_logit_mask = logit_masks[0]
        for mask in logit_masks:
          combined_logit_mask = np.add(combined_logit_mask, mask)

        if sample_index > 7: 
          cal_sgmd.append(combined_logit_mask)
          cal_gt_masks.append(ground_truth)
        else:
          val_sgmd.append(combined_logit_mask)
          val_gt_masks.append(ground_truth)
      
    cal_sgmd = np.array(cal_sgmd)
    cal_gt_masks = np.array(cal_gt_masks)
    val_sgmd = np.array(val_sgmd)
    val_gt_masks = np.array(val_gt_masks)

    def false_negative_rate(pred_masks, true_masks):
        return 1-((pred_masks * true_masks).sum(axis=1).sum(axis=1)/true_masks.sum(axis=1).sum(axis=1)).mean()
    def lamhat_threshold(lam): 
        n = len(cal_sgmd)
        return false_negative_rate(cal_sgmd>=lam, cal_gt_masks) - ((n+1)/n*ALPHA - 1/n)
    lamhat = brentq(lamhat_threshold, -1e10, 1)   


    for sample_index in range(1, 10):
      if sample_index > 7: continue
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

        logit_masks = lora_sam.cached_predict(full_logit_cache_path, full_image_path, full_mask_path, return_logits=True)
        combined_logit_mask = logit_masks[0]
        for mask in logit_masks:
          combined_logit_mask = np.add(combined_logit_mask, mask)
        combined_logit_mask = combined_logit_mask >= lamhat
        logical_or_mask = combined_logit_mask.astype(int)
        print("SHAPE:", logical_or_mask.shape)
        print("MASK:", logical_or_mask)
        show_black_and_white_image(logical_or_mask)

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

    to_append = {
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
    }
    performance.append(to_append)
    print(to_append)
    
  df = pd.DataFrame(performance)

  if not os.path.exists(f"output/lora_metrics/rank_{rank}/model_{model_path}"):
      os.makedirs(f"output/lora_metrics/rank_{rank}/model_{model_path}") 
  df.to_csv(f"output/lora_metrics/rank_{rank}/model_{model_path}/performance_conformal.csv", index=False)