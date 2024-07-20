import os

from PIL import Image
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from segment_anything_utils import SegmentAnythingSession, save_black_and_white_image
from segmentation_metrics import segmentation_metrics, average_dict_values
from lora_inference import LoraSamInference, combine_binary_masks

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

def calculate_uncertainty(matrix):
    matrix = matrix - 1
    matrix = np.maximum(matrix, 0)
    return matrix.sum()

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


for (target_object_name, target_object_colors) in [
    ("ferris_wheel", FERRIS_WHEEL_COLORS),
    ("tree", TREE_COLORS),
    ("roller_coaster", ROLLER_COASTER_COLORS),
    ("carousel", CAROUSEL_COLORS),
]:
    for weather in WEATHER:
        lora_sam = LoraSamInference("./model_checkpoint/sam_vit_b_01ec64.pth", f"finetuned_weights/{target_object_name}/{weather}/lora_rank{RANK}.safetensors", RANK)
        predictions = []
        conformal_predictions = []
        uncertainties = []
        for sample_weather in WEATHER:
            lamhat = calculate_lambda(target_object_name, weather, sample_weather)
            
            for sample in os.listdir(images_path):
                matrix = []
                average_predictions = []
                images_path = f"dataset/{target_object_name}/{sample_weather}/test/images"
                masks_path = f"dataset/{target_object_name}/{sample_weather}/test/masks"
                sample_image_path = os.path.join(images_path, sample)
                sample_mask_path = os.path.join(masks_path, sample)
                sample_prediction_cache_path = f"cache/{weather}_model/{sample_image_path}/prediction.npy"
                sample_logit_cache_path = f"cache/{weather}_model/{sample_image_path}/logit.npy"

                prediction = lora_sam.cached_predict(
                    cached_path=sample_prediction_cache_path, 
                    image_path=sample_image_path, 
                    mask_path=sample_mask_path,
                    verbose=True
                )
                prediction = combine_binary_masks(prediction)
                matrix.append(prediction)

                prediction_png_path = f"results/{target_object_name}_{weather}_rank{RANK}"
                if not os.path.exists(prediction_png_path):
                    os.makedirs(prediction_png_path)
                save_black_and_white_image(prediction, f"{prediction_png_path}/{sample}.png")

                logit_prediction = lora_sam.cached_predict(
                    cached_path=sample_logit_cache_path, 
                    image_path=sample_image_path, 
                    mask_path=sample_mask_path,
                    return_logits=True,
                    verbose=True
                )
                sgmd_prediction = logits_to_sgmd(logit_prediction)
                conformal_prediction = sgmd_prediction >= lamhat

                ground_truth = SegmentAnythingSession.filter_colors(sample_mask_path, [(255, 255, 255)])
                
                predictions.append(segmentation_metrics(ground_truth, prediction))
                conformal_predictions.append(segmentation_metrics(ground_truth, conformal_prediction))


        predictions = pd.DataFrame(predictions)
        predictions.to_csv(f"results/{target_object_name}_{weather}_rank{RANK}.csv", index=False)

        conformal_predictions = pd.DataFrame(conformal_predictions)
        conformal_predictions.to_csv(f"results/{target_object_name}_{weather}_rank{RANK}_conformal.csv", index=False)

        uncertainties = pd.DataFrame(uncertainties)
        uncertainties.to_csv(f"results/{target_object_name}_{weather}_rank{RANK}_uncertainty.csv", index=False)