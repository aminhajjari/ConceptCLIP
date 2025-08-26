"""
ConceptCLIP classification module for MILK10k pipeline
"""
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from transformers import AutoModel, AutoProcessor
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class ConceptCLIPClassifier:
    """ConceptCLIP-based classification for medical images"""
    
    def __init__(self, model_name: str, device: str, classification_config: Dict[str, Any]):
        self.model_name = model_name
        self.device = device
        self.config = classification_config
        self.model = None
        self.processor = None
        self._load_models()
    
    def _load_models(self):
    """Load ConceptCLIP model and processor from cache"""
    try:
        cache_dir = "./models/conceptclip/transformers"
        print(f"Loading ConceptCLIP from cache: {cache_dir}")
        
        # Load model from cache (no token needed)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype="auto",
            local_files_only=True  # Use cached files only
        )
        
        # Load processor from cache
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            local_files_only=True  # Use cached files only
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        
        print(f"✅ ConceptCLIP loaded from cache on {self.device}")
        
    except Exception as e:
        print(f"❌ Error loading cached ConceptCLIP: {e}")
        print("💡 Run: python utils/quick_download_conceptclip.py to download model")
        raise e
    
    def classify_single_image(self, image: np.ndarray, text_prompts: List[str]) -> Dict[str, float]:
        """Classify a single image against text prompts"""
        try:
            # Convert to PIL Image
            if image.size == 0:
                return {}
            
            pil_image = Image.fromarray(image.astype(np.uint8))
            
            # Process inputs
            inputs = self.processor(
                images=pil_image, 
                text=text_prompts,
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = (outputs['logit_scale'] * outputs['image_features'] @ outputs['text_features'].t()).softmax(dim=-1)[0]
            
            # Convert to probabilities
            disease_names = [prompt.split(' showing ')[-1] for prompt in text_prompts]
            probabilities = {disease_names[i]: float(logits[i]) for i in range(len(disease_names))}
            
            return probabilities
            
        except Exception as e:
            print(f"Classification error for single image: {e}")
            return {}
    
    def classify_segmented_outputs(self, segmented_outputs: Dict[str, np.ndarray], 
                                  text_prompts: List[str]) -> Dict[str, Dict[str, float]]:
        """Classify multiple segmented outputs"""
        results = {}
        
        for output_type, seg_image in segmented_outputs.items():
            if seg_image is not None and seg_image.size > 0:
                probabilities = self.classify_single_image(seg_image, text_prompts)
                if probabilities:
                    results[output_type] = probabilities
        
        return results
    
    def ensemble_predictions(self, classification_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Ensemble multiple predictions using weighted averaging"""
        if not classification_results:
            return {}
        
        # Get ensemble weights from config
        weights = self.config.get('ensemble_weights', {})
        
        # Get all disease names
        disease_names = list(next(iter(classification_results.values())).keys())
        
        # Weighted average
        final_probs = {}
        for disease in disease_names:
            weighted_sum = 0
            total_weight = 0
            
            for output_type, probs in classification_results.items():
                if output_type in weights and disease in probs:
                    weight = weights[output_type]
                    weighted_sum += weight * probs[disease]
                    total_weight += weight
                elif output_type not in weights:
                    # Default weight for unknown output types
                    default_weight = 0.1
                    weighted_sum += default_weight * probs[disease]
                    total_weight += default_weight
            
            final_probs[disease] = weighted_sum / total_weight if total_weight > 0 else 0
        
        return final_probs
    
    def get_top_predictions(self, probabilities: Dict[str, float], top_k: int = 3) -> List[tuple]:
        """Get top-k predictions with confidence scores"""
        sorted_preds = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        return sorted_preds[:top_k]
    
    def classify_with_confidence_filtering(self, segmented_outputs: Dict[str, np.ndarray], 
                                         text_prompts: List[str]) -> Dict:
        """Classify with confidence-based filtering"""
        # Get individual predictions
        classification_results = self.classify_segmented_outputs(segmented_outputs, text_prompts)
        
        if not classification_results:
            return {
                'ensemble_prediction': {},
                'individual_predictions': {},
                'confidence_metrics': {},
                'filtered_predictions': {}
            }
        
        # Ensemble predictions
        ensemble_probs = self.ensemble_predictions(classification_results)
        
        # Apply confidence filtering
        confidence_threshold = self.config.get('confidence_threshold', 0.1)
        filtered_probs = {
            disease: prob for disease, prob in ensemble_probs.items()
            if prob >= confidence_threshold
        }
        
        # Calculate confidence metrics
        confidence_metrics = self._calculate_confidence_metrics(classification_results, ensemble_probs)
        
        return {
            'ensemble_prediction': ensemble_probs,
            'individual_predictions': classification_results,
            'confidence_metrics': confidence_metrics,
            'filtered_predictions': filtered_probs
        }
    
    def _calculate_confidence_metrics(self, individual_results: Dict, ensemble_probs: Dict) -> Dict:
        """Calculate various confidence metrics"""
        if not individual_results or not ensemble_probs:
            return {}
        
        # Calculate prediction consistency across different segmentation outputs
        disease_names = list(ensemble_probs.keys())
        consistencies = {}
        
        for disease in disease_names:
            predictions = [probs.get(disease, 0) for probs in individual_results.values()]
            if predictions:
                consistency = 1.0 - np.std(predictions) / (np.mean(predictions) + 1e-6)
                consistencies[disease] = max(0, consistency)  # Ensure non-negative
        
        # Overall confidence metrics
        max_prob = max(ensemble_probs.values()) if ensemble_probs else 0
        entropy = self._calculate_entropy(list(ensemble_probs.values()))
        
        return {
            'max_confidence': max_prob,
            'prediction_entropy': entropy,
            'prediction_consistency': consistencies,
            'average_consistency': np.mean(list(consistencies.values())) if consistencies else 0
        }
    
    def _calculate_entropy(self, probabilities: List[float]) -> float:
        """Calculate entropy of probability distribution"""
        probs = np.array(probabilities)
        probs = probs[probs > 0]  # Remove zero probabilities
        if len(probs) == 0:
            return 0
        return -np.sum(probs * np.log(probs + 1e-10))
    
    def batch_classify(self, images: List[np.ndarray], text_prompts: List[str]) -> List[Dict[str, float]]:
        """Classify multiple images in batch"""
        results = []
        
        for image in images:
            probabilities = self.classify_single_image(image, text_prompts)
            results.append(probabilities)
        
        return results
    
    def get_prediction_summary(self, classification_result: Dict) -> Dict:
        """Get a summary of classification results"""
        ensemble_probs = classification_result.get('ensemble_prediction', {})
        confidence_metrics = classification_result.get('confidence_metrics', {})
        
        if not ensemble_probs:
            return {'error': 'No predictions available'}
        
        # Top prediction
        top_prediction = max(ensemble_probs.items(), key=lambda x: x[1])
        
        # Get top-3 predictions
        top_3 = self.get_top_predictions(ensemble_probs, top_k=3)
        
        return {
            'predicted_class': top_prediction[0],
            'confidence': top_prediction[1],
            'top_3_predictions': top_3,
            'max_confidence': confidence_metrics.get('max_confidence', 0),
            'entropy': confidence_metrics.get('prediction_entropy', 0),
            'consistency': confidence_metrics.get('average_consistency', 0),
            'num_outputs_used': len(classification_result.get('individual_predictions', {}))
        }

# ==================== UTILITY FUNCTIONS ====================

def validate_text_prompts(text_prompts: List[str]) -> bool:
    """Validate text prompts format"""
    if not isinstance(text_prompts, list) or len(text_prompts) == 0:
        return False
    
    for prompt in text_prompts:
        if not isinstance(prompt, str) or len(prompt.strip()) == 0:
            return False
    
    return True

def create_medical_prompts(conditions: List[str]) -> List[str]:
    """Create medical text prompts from condition names"""
    prompts = []
    for condition in conditions:
        prompt = f"a medical image showing {condition}"
        prompts.append(prompt)
    
    return prompts

def evaluate_classification_performance(predictions: List[Dict], ground_truth: List[str]) -> Dict:
    """Evaluate classification performance metrics"""
    if len(predictions) != len(ground_truth):
        raise ValueError("Number of predictions must match number of ground truth labels")
    
    correct = 0
    total = len(predictions)
    
    # Per-class metrics
    class_correct = {}
    class_total = {}
    
    for pred_dict, true_label in zip(predictions, ground_truth):
        if not pred_dict:
            continue
        
        predicted_label = max(pred_dict.items(), key=lambda x: x[1])[0]
        
        # Overall accuracy
        if predicted_label == true_label:
            correct += 1
        
        # Per-class tracking
        if true_label not in class_total:
            class_total[true_label] = 0
            class_correct[true_label] = 0
        
        class_total[true_label] += 1
        if predicted_label == true_label:
            class_correct[true_label] += 1
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    
    class_accuracies = {}
    for class_name in class_total:
        if class_total[class_name] > 0:
            class_accuracies[class_name] = class_correct[class_name] / class_total[class_name]
    
    return {
        'overall_accuracy': accuracy,
        'correct_predictions': correct,
        'total_predictions': total,
        'class_accuracies': class_accuracies,
        'average_class_accuracy': np.mean(list(class_accuracies.values())) if class_accuracies else 0
    }