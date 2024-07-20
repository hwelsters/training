from typing import List

from lora_inference import LoraSamInference, combine_binary_masks

class MultiloraPredictor:
    def __init__(self, target_objects: List[str], model_checkpoint, weather, rank = 8):
        self.models = {}
        for target_object in target_objects:
            self.models[target_object] = LoraSamInference(model_checkpoint, f"finetuned_weights/{target_object}/lora_rank{rank}.safetensors", rank)

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