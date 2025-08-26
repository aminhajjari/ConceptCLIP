"""
MILK10k Pipeline Configuration Settings
"""
import os
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

# ==================== PATH CONFIGURATION ====================

# Your dataset paths - modify these according to your setup
DATASET_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/MILK10k_Training_Input"
GROUNDTRUTH_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/groundtruth.csv"
OUTPUT_PATH = "/project/def-arashmoh/shahab33/XAI/outputs"

# ==================== MODEL CONFIGURATION ====================

# SAM2 Model Configuration
SAM2_MODEL = "facebook/sam2-hiera-large"

# ConceptCLIP Model Configuration
CONCEPTCLIP_MODEL = "JerrryNie/ConceptCLIP"
CONCEPTCLIP_CACHE_DIR = "./models/conceptclip/transformers"
USE_CACHED_CONCEPTCLIP = True

# Device Configuration
DEVICE_PREFERENCE = "cuda"  # "cuda" or "cpu"

# ==================== DOMAIN CONFIGURATION ====================

@dataclass
class MedicalDomain:
    """Configuration for MILK10k medical imaging domain"""
    name: str
    image_extensions: List[str]
    text_prompts: List[str]
    label_mappings: Dict[str, str]
    preprocessing_params: Dict
    segmentation_strategy: str

# MILK10k Medical Domain Configuration
MILK10K_DOMAIN = MedicalDomain(
    name="milk10k",
    image_extensions=['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.dcm', '.dicom', '.nii', '.nii.gz'],
    text_prompts=[
        'a medical image showing normal tissue',
        'a medical image showing abnormal pathology',
        'a medical image showing inflammatory lesion',
        'a medical image showing neoplastic lesion',
        'a medical image showing degenerative changes',
        'a medical image showing infectious disease',
        'a medical image showing vascular pathology',
        'a medical image showing metabolic disorder',
        'a medical image showing congenital abnormality',
        'a medical image showing traumatic injury'
    ],
    label_mappings={
        'NORMAL': 'normal tissue',
        'ABNORMAL': 'abnormal pathology',
        'INFLAMMATORY': 'inflammatory lesion',
        'NEOPLASTIC': 'neoplastic lesion',
        'DEGENERATIVE': 'degenerative changes',
        'INFECTIOUS': 'infectious disease',
        'VASCULAR': 'vascular pathology',
        'METABOLIC': 'metabolic disorder',
        'CONGENITAL': 'congenital abnormality',
        'TRAUMATIC': 'traumatic injury'
    },
    preprocessing_params={
        'normalize': True, 
        'enhance_contrast': True,
        'clahe_clip_limit': 2.0,
        'clahe_tile_grid_size': (8, 8)
    },
    segmentation_strategy='adaptive'
)

# ==================== PROCESSING CONFIGURATION ====================

# Segmentation Parameters
SEGMENTATION_CONFIG = {
    'multimask_output': True,
    'roi_padding': 20,
    'max_roi_points': 3
}

# Classification Parameters
CLASSIFICATION_CONFIG = {
    'ensemble_weights': {
        'colored_overlay': 0.3,
        'contour': 0.2,
        'cropped': 0.25,
        'masked_only': 0.15,
        'side_by_side': 0.1
    },
    'confidence_threshold': 0.1
}

# Output Configuration
OUTPUT_STRUCTURE = {
    'segmented': 'segmented',
    'segmented_for_conceptclip': 'segmented_for_conceptclip',
    'classifications': 'classifications',
    'visualizations': 'visualizations',
    'reports': 'reports'
}

# ==================== VISUALIZATION CONFIGURATION ====================

PLOT_CONFIG = {
    'figsize': (15, 12),
    'dpi': 300,
    'style': 'default',
    'color_palette': 'viridis'
}

# ==================== UTILITY FUNCTIONS ====================

def get_device():
    """Get the appropriate device for computation"""
    import torch
    if DEVICE_PREFERENCE == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def create_output_directories(base_path: Path):
    """Create all necessary output directories"""
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    for dir_name in OUTPUT_STRUCTURE.values():
        (base_path / dir_name).mkdir(exist_ok=True)
    
    return base_path

def validate_paths():
    """Validate that all required paths exist"""
    dataset_path = Path(DATASET_PATH)
    groundtruth_path = Path(GROUNDTRUTH_PATH)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    if not groundtruth_path.exists():
        print(f"Warning: Ground truth file not found: {groundtruth_path}")
        return False
    
    return True