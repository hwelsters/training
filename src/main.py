import os

import airsim
from colorama import Fore, Back, Style
import pandas

from caching import is_numpy_array_cached, cache_numpy_array, load_cached_numpy_array
from log_utils import log
from multirotor_session import MultirotorSession
from segment_anything_utils import SegmentAnythingSession, save_black_and_white_image
from segmentation_metrics import segmentation_classification_report, segmentation_dice_score, segmentation_jaccard_score

# ==============================
# Segment images
# ==============================
segmentAnything = SegmentAnythingSession(model_name="vit_b")

ferris_wheel_colors = [(45,  157, 177), (219, 103, 127), (112, 158, 139), (117, 116, 121)]

performance = []
for image_path, text_prompt, colors in [
    ("images/normal", "ferris wheel", ferris_wheel_colors),

    ("images/fog-0.1", "ferris wheel", ferris_wheel_colors),
    ("images/fog-0.2", "ferris wheel", ferris_wheel_colors),
    ("images/fog-0.4", "ferris wheel", ferris_wheel_colors),

    ("images/rain-0.2", "ferris wheel", ferris_wheel_colors),
    ("images/rain-0.4", "ferris wheel", ferris_wheel_colors),
    ("images/rain-0.8", "ferris wheel", ferris_wheel_colors),
    
    ("images/snow-0.2", "ferris wheel", ferris_wheel_colors),
    ("images/snow-0.4", "ferris wheel", ferris_wheel_colors),
    ("images/snow-0.8", "ferris wheel", ferris_wheel_colors),

    ("images/dust-0.1", "ferris wheel", ferris_wheel_colors),
    ("images/dust-0.2", "ferris wheel", ferris_wheel_colors),
    ("images/dust-0.4", "ferris wheel", ferris_wheel_colors),

    ("images/mapleleaf-0.1", "ferris wheel", ferris_wheel_colors),
    ("images/mapleleaf-0.2", "ferris wheel", ferris_wheel_colors),
    ("images/mapleleaf-0.4", "ferris wheel", ferris_wheel_colors),

]:
    for sample_index in range(1, 10):
        average_accuracy = 0
        average_precision_background = 0
        average_recall_background = 0
        average_f1_background = 0
        average_precision_foreground = 0
        average_recall_foreground = 0
        average_f1_foreground = 0
        average_dice_score = 0
        average_jaccard_score = 0

        for index in range(0, 10):
            input_image_path = image_path + f"/sample_{sample_index}/{index}/Scene.png"
            ground_truth_path = image_path + f"/sample_{sample_index}/{index}/Segmentation.png"

            cache_prediction_path = input_image_path.replace("images", "cache/predictions").replace("Scene.png", f"prediction.npy")

            prediction = segmentAnything.cached_prompt_predict(input_image_path, cache_prediction_path, text_prompt)
            save_black_and_white_image(prediction, ground_truth_path.replace("Segmentation.png", "Output.png"))

            ground_truth = segmentAnything.filter_colors(ground_truth_path, colors)
            save_black_and_white_image(ground_truth, ground_truth_path.replace("Segmentation.png", "Masks.png"))

            classification_report = segmentation_classification_report(ground_truth, prediction)
            dice_score = segmentation_dice_score(ground_truth, prediction)
            jaccard_score = segmentation_jaccard_score(ground_truth, prediction)

            average_accuracy += classification_report.get("accuracy", 0) 
            average_precision_background += classification_report["background"].get("precision", 0)
            average_recall_background += classification_report["background"].get("recall", 0)
            average_f1_background += classification_report["background"].get("f1-score", 0)
            average_precision_foreground += classification_report["foreground"].get("precision", 0)
            average_recall_foreground += classification_report["foreground"].get("recall", 0)
            average_f1_foreground += classification_report["foreground"].get("f1-score", 0)
            average_dice_score += dice_score
            average_jaccard_score += jaccard_score

        performance_results = {
            "ground_truth_path": ground_truth_path,
            "image_path": image_path,
            "text_prompt": text_prompt,

            # === Classification report ===
            "Average accuracy": average_accuracy / 10,
            "Average precision (background)": average_precision_background / 10,
            "Average recall (background)": average_recall_background / 10,
            "Average f1-score (background)": average_f1_background / 10,
            "Average precision (foreground)": average_precision_foreground / 10,
            "Average recall (foreground)": average_recall_foreground / 10,
            "Average f1-score (foreground)": average_f1_foreground / 10,

            # === Dice score ===
            "Average dice_score": average_dice_score / 10,

            # === Jaccaard score ===
            "Average jaccard_score": average_jaccard_score / 10,

        }
        performance.append(performance_results)
        performance_df = pandas.DataFrame(performance)
        performance_df.to_csv("performance-path.csv")