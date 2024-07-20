import os

from colorama import Fore, Back, Style
from PIL import Image
import numpy as np
import torch

from caching import is_numpy_array_cached, cache_numpy_array, load_cached_numpy_array
from lora_k.segment_anything import build_sam_vit_b, SamPredictor
from lora_k.lora import LoRA_sam
from lora_k.utils import get_bounding_box
from log_utils import log

def combine_binary_masks(masks):
    combined_mask = np.zeros(masks[0].shape)
    for mask in masks:
        combined_mask = np.logical_or(combined_mask, mask)
    return combined_mask

class LoraSamInference:
    def __init__(self, model_checkpoint: str, finetuned_weights: str, rank:int = 4):
        self.sam = build_sam_vit_b(model_checkpoint)
        self.rank = rank

        self.sam_lora = LoRA_sam(self.sam, self.rank)
        self.sam_lora.load_lora_parameters(finetuned_weights)

        self.model = self.sam_lora.sam

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

    def cached_predict(self, cached_path: str, image_path: str, mask_path: str, return_logits: bool = False, verbose: bool = False):
        if not is_numpy_array_cached(cached_path):
            if verbose: log(f"{Fore.RED}CACHE MISS!{Fore.RESET} prediction from:", cached_path)
            masks = self.predict(image_path, mask_path, return_logits=return_logits)

            if verbose: log("CACHING prediction to:", cached_path)
            cache_numpy_array(masks, cached_path)
            
            return masks
        else: 
            if verbose: log(f"{Fore.GREEN}CACHE HIT!{Fore.RESET} prediction from:", cached_path)
            return load_cached_numpy_array(cached_path)

    def predict(self, image_path: str, mask_path: str, return_logits: bool = False):
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = np.array(image)

        # Get bounding box.
        mask = Image.open(mask_path)
        mask = mask.convert('1')
        mask = np.array(mask)

        box = [0, 0, 256, 144]

        predictor = SamPredictor(self.model)
        predictor.set_image(image)
        masks, iou_pred, low_res_iou = predictor.predict(box=np.array([box]), multimask_output=False, return_logits=return_logits)

        return masks