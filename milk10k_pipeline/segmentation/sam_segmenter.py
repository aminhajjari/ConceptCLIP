"""
SAM2 segmentation module for MILK10k pipeline
"""
import torch
import numpy as np
import cv2
from typing import Tuple, Dict, Any
from sam2.sam2_image_predictor import SAM2ImagePredictor
from skimage import filters, morphology, measure
import warnings
warnings.filterwarnings('ignore')

class SAMSegmenter:
    """SAM2-based segmentation for medical images"""
    
    def __init__(self, model_name: str, device: str, segmentation_config: Dict[str, Any]):
        self.model_name = model_name
        self.device = device
        self.config = segmentation_config
        self.predictor = None
        self._load_model()
    
    def _load_model(self):
        """Load SAM2 model"""
        try:
            print(f"Loading SAM2 model: {self.model_name}")
            self.predictor = SAM2ImagePredictor.from_pretrained(self.model_name)
            print(f"SAM2 loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading SAM2: {e}")
            raise e
    
    def segment_image(self, image: np.ndarray, strategy: str = 'adaptive') -> Tuple[np.ndarray, float]:
        """Main segmentation function with different strategies"""
        if strategy == 'adaptive':
            return self._adaptive_segmentation(image)
        elif strategy == 'center_point':
            return self._center_point_segmentation(image)
        elif strategy == 'multi_point':
            return self._multi_point_segmentation(image)
        else:
            raise ValueError(f"Unknown segmentation strategy: {strategy}")
    
    def _adaptive_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Adaptive segmentation based on image content"""
        h, w = image.shape[:2]
        
        # Find regions of interest
        roi_points = self._find_roi_points(image)
        point_labels = np.ones(len(roi_points))
        
        # SAM2 segmentation
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(image)
            masks, scores, logits = self.predictor.predict(
                point_coords=roi_points,
                point_labels=point_labels,
                box=None,
                multimask_output=self.config.get('multimask_output', True)
            )
        
        # Select best mask
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx].astype(np.uint8)
        confidence = float(scores[best_idx])
        
        # Post-process mask
        mask = self._post_process_mask(mask, image.shape[:2])
        
        return mask, confidence
    
    def _center_point_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Simple center-point segmentation"""
        h, w = image.shape[:2]
        center_point = np.array([[w // 2, h // 2]])
        point_labels = np.array([1])
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(image)
            masks, scores, logits = self.predictor.predict(
                point_coords=center_point,
                point_labels=point_labels,
                box=None,
                multimask_output=self.config.get('multimask_output', True)
            )
        
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx].astype(np.uint8)
        confidence = float(scores[best_idx])
        
        return mask, confidence
    
    def _multi_point_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Multi-point segmentation using grid points"""
        h, w = image.shape[:2]
        
        # Create grid of points
        grid_size = 3
        points = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = int((j + 1) * w / (grid_size + 1))
                y = int((i + 1) * h / (grid_size + 1))
                points.append([x, y])
        
        points = np.array(points)
        point_labels = np.ones(len(points))
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(image)
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=point_labels,
                box=None,
                multimask_output=self.config.get('multimask_output', True)
            )
        
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx].astype(np.uint8)
        confidence = float(scores[best_idx])
        
        return mask, confidence
    
    def _find_roi_points(self, image: np.ndarray) -> np.ndarray:
        """Find regions of interest for adaptive segmentation"""
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur to reduce noise
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Find high-contrast regions using Otsu thresholding
        try:
            thresh = filters.threshold_otsu(gray_blurred)
            binary = gray_blurred > thresh
        except:
            # Fallback if Otsu fails
            binary = gray_blurred > np.mean(gray_blurred)
        
        # Clean up binary image
        binary = morphology.remove_small_objects(binary, min_size=100)
        binary = morphology.remove_small_holes(binary, area_threshold=100)
        
        # Find connected components
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)
        
        # Select regions based on size and location
        roi_points = []
        max_roi_points = self.config.get('max_roi_points', 3)
        
        # Sort regions by area (largest first)
        regions = sorted(regions, key=lambda x: x.area, reverse=True)
        
        for region in regions[:max_roi_points]:
            # Check if region is reasonable size
            if region.area > 50:  # Minimum area threshold
                y, x = region.centroid
                roi_points.append([int(x), int(y)])
        
        # If no regions found, use fallback points
        if not roi_points:
            h, w = image.shape[:2]
            roi_points = [
                [w // 2, h // 2],  # Center
                [w // 3, h // 3],  # Upper left quadrant
                [2 * w // 3, 2 * h // 3]  # Lower right quadrant
            ]
        
        return np.array(roi_points)
    
    def _post_process_mask(self, mask: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """Post-process segmentation mask"""
        # Remove small isolated components
        mask = morphology.remove_small_objects(mask.astype(bool), min_size=50)
        
        # Fill small holes
        mask = morphology.remove_small_holes(mask, area_threshold=50)
        
        # Smooth the mask boundaries
        mask = morphology.binary_closing(mask, morphology.disk(2))
        
        return mask.astype(np.uint8)
    
    def create_segmented_outputs(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
        """Create multiple segmented outputs for classification"""
        outputs = {}
        padding = self.config.get('roi_padding', 20)
        
        # 1. Original image with colored overlay
        color = (255, 0, 0)  # Red highlight
        overlay = image.copy()
        overlay[mask == 1] = color
        colored_overlay = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        outputs['colored_overlay'] = colored_overlay
        
        # 2. Contour highlighting
        contour_image = image.copy()
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)
        outputs['contour'] = contour_image
        
        # 3. Cropped to bounding box
        coords = np.where(mask == 1)
        if len(coords[0]) > 0:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # Add padding
            h, w = image.shape[:2]
            y_min = max(0, y_min - padding)
            y_max = min(h, y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(w, x_max + padding)
            
            cropped = image[y_min:y_max, x_min:x_max]
            if cropped.size > 0:
                outputs['cropped'] = cropped
        
        # 4. Masked region only (with black background)
        masked_only = image.copy()
        masked_only[mask == 0] = [0, 0, 0]
        outputs['masked_only'] = masked_only
        
        # 5. Side-by-side comparison
        if 'cropped' in outputs and outputs['cropped'].size > 0:
            # Resize cropped to match original height for side-by-side
            h_orig = image.shape[0]
            cropped_shape = outputs['cropped'].shape
            if cropped_shape[0] > 0 and cropped_shape[1] > 0:
                aspect_ratio = cropped_shape[1] / cropped_shape[0]
                new_width = int(h_orig * aspect_ratio)
                cropped_resized = cv2.resize(outputs['cropped'], (new_width, h_orig))
                side_by_side = np.hstack([image, cropped_resized])
                outputs['side_by_side'] = side_by_side
        
        return outputs
    
    def get_segmentation_quality_metrics(self, mask: np.ndarray) -> Dict[str, float]:
        """Calculate quality metrics for segmentation"""
        total_pixels = mask.size
        segmented_pixels = np.sum(mask)
        
        # Calculate basic metrics
        coverage = segmented_pixels / total_pixels
        
        # Calculate compactness (perimeter^2 / area)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perimeter = cv2.arcLength(contours[0], True)
            area = cv2.contourArea(contours[0])
            compactness = (perimeter ** 2) / (4 * np.pi * area + 1e-6)
        else:
            compactness = 0
        
        return {
            'coverage': coverage,
            'compactness': compactness,
            'area': float(segmented_pixels),
            'perimeter': perimeter if contours else 0
        }

# ==================== UTILITY FUNCTIONS ====================

def validate_mask(mask: np.ndarray) -> bool:
    """Validate segmentation mask"""
    if mask is None or mask.size == 0:
        return False
    
    # Check if mask contains valid values (0 or 1)
    unique_vals = np.unique(mask)
    if not all(val in [0, 1] for val in unique_vals):
        return False
    
    # Check if mask has any segmented regions
    if np.sum(mask) == 0:
        return False
    
    return True