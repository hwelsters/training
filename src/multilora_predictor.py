import numpy as np
from typing import List

from lora_inference import LoraSamInference, combine_binary_masks

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
    
    def cached_uncertainties(self, cache_path, image_path, mask_path):
        predictions = self.cached_predict(cache_path, image_path, mask_path)
        return MultiloraPredictor.calculate_uncertainty(predictions.values())
    
    def calculate_uncertainty(matrixes: List[np.array]):
        # Convert false to 0 and true to 1
        size = len(matrixes)

        matrixes = [matrix.astype(int) for matrix in matrixes]
        print(matrixes)

        np_matrixes = np.array(matrixes)

        # element wise sum of matrixes
        sum_matrix = np.sum(np_matrixes, axis=0)
        print(sum_matrix)

        # element wise minus one
        minus_one_matrix = sum_matrix - 1

        # element wise make sure at least 0
        zero_matrix = np.maximum(minus_one_matrix, 0)

        # total mean of all elements
        total_mean = np.mean(zero_matrix)

        return total_mean / size