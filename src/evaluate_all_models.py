import os
import math
from typing import List

from PIL import Image
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from segment_anything_utils import SegmentAnythingSession, save_black_and_white_image, save_image
from segmentation_metrics import segmentation_metrics, average_dict_values
from lora_inference import LoraSamInference, combine_binary_masks
from multilora_predictor import MultiloraPredictor

# ==============================
# COLORS
# ==============================
FERRIS_WHEEL_COLORS = [(45,  157, 177), (219, 103, 127), (112, 158, 139), (117, 116, 121)]
TREE_COLORS = [(95, 224, 39)]
ROLLER_COASTER_COLORS = [(108, 116, 224)]
CAROUSEL_COLORS = [(117, 93, 91)]

MAX_TRAINING_SET_INDEX = 12
MIN_TEST_SET_INDEX = 17

MIN_INDEX = 5
MAX_INDEX = 6
OUTPUT_DIR = "dataset"

TRAINING_SET_SIZE = 10
TEST_SET_SIZE = 10

RANK=8
ALPHA=0.2

WEATHER = [
    "dust-0.4",
    "mapleleaf-0.4",
    "normal",
]

def logits_to_sgmd(logits):
    return 1/(1+np.exp(-logits))

def calculate_lambda(target_object_name, weather, sample_weather):
    masks_path = f"dataset/{target_object_name}/{sample_weather}/train/masks"
    image_path = f"dataset/{target_object_name}/{sample_weather}/train/images"

    cal_sgmd = []
    cal_gt_masks = []
    for sample in os.listdir(masks_path):
        full_masks_path = os.path.join(masks_path, sample)
        full_image_path = os.path.join(image_path, sample)

        sample_logit_cache_path = f"cache/{weather}_model/{full_image_path}/logit.npy"
        logit_masks = lora_sam.cached_predict(sample_logit_cache_path, full_image_path, full_masks_path, return_logits=True)
        ground_truth = SegmentAnythingSession.filter_colors(full_masks_path, [(255, 255, 255)])
        sgmd_masks = logits_to_sgmd(logit_masks)
        cal_sgmd.append(sgmd_masks)
        cal_gt_masks.append(ground_truth)
    
    cal_sgmd = np.array(cal_sgmd)
    cal_gt_masks = np.array(cal_gt_masks)

    def false_negative_rate(pred_masks, true_masks):
        return 1-((pred_masks * true_masks).sum(axis=1).sum(axis=1)/true_masks.sum(axis=1).sum(axis=1)).mean()
    def lamhat_threshold(lam): 
        n = len(cal_sgmd)
        return false_negative_rate(cal_sgmd>=lam, cal_gt_masks) - ((n+1)/n*ALPHA - 1/n)
    lamhat = brentq(lamhat_threshold, -1e10, 1)   
    return lamhat


uncertainties = []
for weather in WEATHER:
    multilora_predictor = MultiloraPredictor(["ferris_wheel", "tree", "roller_coaster", "carousel"], f"model_checkpoint/sam_vit_b_01ec64.pth", weather, rank=RANK)
    TRAIN_IMAGE_PATH = 'dataset/train/images'
    TEST_IMAGE_PATH = 'dataset/test/images'
    TRAIN_MASK_PATH = 'dataset/train/masks'
    TEST_MASK_PATH = 'dataset/test/masks'

    for sample in os.listdir(TRAIN_IMAGE_PATH):
        full_masks_path = os.path.join(TRAIN_MASK_PATH, sample)
        full_image_path = os.path.join(TRAIN_IMAGE_PATH, sample)

        full_prediction_cache_path = f"cache/{weather}_model/{full_image_path}/prediction.npy"
        
        uncertainty_matrix = multilora_predictor.cached_average_uncertainties(full_prediction_cache_path, full_image_path, full_masks_path)
        save_image(uncertainty_matrix, f"uncertainty/{weather}_model/{full_image_path}/uncertainty.png")

        uncertainties.append({
            "path": full_prediction_cache_path,
            "uncertainty": multilora_predictor.cached_average_uncertainties(full_prediction_cache_path, full_image_path, full_masks_path)
        })

uncertainties_df = pd.DataFrame(uncertainties)
uncertainties_df.to_csv("uncertainties.csv", index=False)