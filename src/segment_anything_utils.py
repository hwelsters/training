import time

from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image

import cv2
from colorama import Fore, Back, Style
import matplotlib.pyplot as plt
import numpy as np
import torch

from caching import is_numpy_array_cached, cache_numpy_array, load_cached_numpy_array
from edcr_drone_config import SEGMENTANYTHING_MODEL_CHECKPOINTS
from log_utils import log

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")


# ==============================
# Helper methods
# ==============================
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   
 
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def show_black_and_white_image(image):
    plt.pcolor(image, cmap='gray', vmin=0, vmax=1)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def save_black_and_white_image(image, path):
    # Set black and white color map
    plt.pcolor(image, cmap='gray', vmin=0, vmax=1)
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


class SegmentAnythingSession:
    def __init__(self, model_name = "default"):
        if model_name not in SEGMENTANYTHING_MODEL_CHECKPOINTS:
            raise ValueError(f"""Model name {model_name} not found in SEGMENTANYTHING_MODEL_CHECKPOINTS
                            Available models: {SEGMENTANYTHING_MODEL_CHECKPOINTS.keys()}
                            To add a new model, download it to ./model_checkpoints and add it to SEGMENTANYTHING_MODEL_CHECKPOINTS in edcr_drone_config.py""")

        path_to_checkpoint = SEGMENTANYTHING_MODEL_CHECKPOINTS[model_name]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = sam_model_registry[model_name](path_to_checkpoint)
        self.model.to(device)

        self.predictor = SamPredictor(self.model)
        self.mask_generator = SamAutomaticMaskGenerator(self.model)

    # ==============================
    # Cached methods
    # ==============================
    def cached_prompt_predict(self, image_path, cache_prediction_path, text_prompt):
        # @info This replace function could potentially cause problems if the path contains multiple occurences of the word "images".
        if not is_numpy_array_cached(cache_prediction_path):
            log(f"{Fore.RED}CACHE MISS!{Fore.RESET} prediction from:", cache_prediction_path)
            cache_numpy_array(self.prompt_predict(image_path, text_prompt), cache_prediction_path)
            log("CACHING prediction to:", cache_prediction_path)
        else: log(f"{Fore.GREEN}CACHE HIT!{Fore.RESET} prediction from:", cache_prediction_path)
        return load_cached_numpy_array(cache_prediction_path)

    # ==============================
    # Convenience methods
    # ==============================
    def predict_image_at_path(self, path, x: int, y: int):
        image = self.read_image_at_path(path)
        # return self.predict(image, x, y,)
        return self.auto_predict(image)

    def read_image_at_path(self, path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def predict(self, image, x: int, y: int):
        self.predictor.set_image(image)

        input_point = np.array([[x, y]])
        input_label = np.array([1])

        masks, scores, logits = self.predictor.predict(
            input_point, input_label, 
            multimask_output=True,
        )

        self.show_masks(masks, scores, image, input_point, input_label)

        return {
            "masks": masks,
            "scores": scores,
            "logits": logits,
        }
    
    def filter_colors(image_path, colors):
        """
        Returns 2D numpy array of 0s and 1s
        """
        ground_truth_pil = Image.open(image_path).convert("RGB")
        
        # Resize to 256x144
        ground_truth_pil = ground_truth_pil.resize((256, 144), Image.NEAREST)

        ground_truth_pil = np.asarray(ground_truth_pil)

        # Iterate through ground truth pil and if color is in ground_truth_colors, set to 1, else 0.
        # It is a 2x2 image/
        ground_truth_pil = np.array([[
            1 if tuple(pixel) in colors else 0
            for pixel in row
        ] for row in ground_truth_pil])
        
        return ground_truth_pil

    def prompt_predict(self, image_path, text_prompt):
        """
        Returns 2D numpy array of 0s and 1s
        """
        log("LOADING:", image_path)
        image_pil = Image.open(image_path).convert("RGB")

        # Get prediction
        log("LANG SAM Prediction:", image_path)
        masks, boxes, phrases, logits = self.lang_sam.predict(image_pil, text_prompt) 
        combined_masks = np.zeros(np.asarray(image_pil).shape[0:2])
        for mask in masks:
            combined_masks = np.logical_or(combined_masks, mask)
        combined_masks = np.array(combined_masks).astype(np.int32)

        log("COMPLETE PREDICTION:", image_path)
        return combined_masks

    def auto_predict(self, image):
        masks = self.mask_generator.generate(image)

        plt.figure(figsize=(20,20))
        plt.imshow(image)
        show_anns(masks)
        plt.axis('off')
        plt.show() 

        return {
            "masks": masks
        }

    
    # ==============================
    # Visualization methods
    # ==============================
    def show_masks(self, masks, scores, image, input_point, input_label):
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            show_mask(mask, plt.gca())
            show_points(input_point, input_label, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()  