import numpy as np
from typing import List

from lora_inference import LoraSamInference, combine_binary_masks
from segment_anything_utils import save_black_and_white_image

class MultiloraPredictor:
    def __init__(self, target_objects: List[str], model_checkpoint, weather, rank = 8):
        self.models = {}
        for target_object in target_objects:
            self.models[target_object] = LoraSamInference(model_checkpoint, f"finetuned_weights/{target_object}/{weather}/lora_rank{rank}.safetensors", rank)

    def predict(self, image_path, mask_path):
        outputs = {}
        for target_object, model in self.models.items():
            outputs[target_object] = model.predict(image_path, mask_path)
        return outputs
    
    def cached_predict(self, cache_path, image_path, mask_path):
        outputs = {}
        for target_object, model in self.models.items():
            outputs[target_object] = model.cached_predict(cache_path, image_path, mask_path)
        return outputs
    
    def cached_average_uncertainties(self, cache_path, image_path, mask_path):
        predictions = self.cached_predict(cache_path, image_path, mask_path)
        return MultiloraPredictor.calculate_average_uncertainty(predictions.values())
    
    def cached_uncertainty(self, cache_path, image_path, mask_path):
        predictions = self.cached_predict(cache_path, image_path, mask_path)
        return MultiloraPredictor.calculate_uncertainty(predictions.values())
    
    def calculate_uncertainty(matrixes: List[np.array]):
        matrixes = [matrix.astype(int) for matrix in matrixes] # Convert false to 0 and true to 1
        np_matrixes = np.array(matrixes)
        sum_matrix = np.sum(np_matrixes, axis=0) # element wise sum of matrixes
        return sum_matrix[0]

    def calculate_average_uncertainty(matrixes: List[np.array]):
        uncertainty_matrix = MultiloraPredictor.calculate_uncertainty(matrixes)
        total_mean = np.mean(uncertainty_matrix)
        return total_mean