# milk10k_pipeline/__init__.py
"""
MILK10k Medical Image Segmentation and Classification Pipeline

A comprehensive pipeline for processing medical images using SAM2 segmentation
and ConceptCLIP classification on the MILK10k dataset.
"""

__version__ = "1.0.0"
__author__ = "MILK10k Pipeline Team"

from core.pipeline import MILK10kPipeline
from .config.settings import MILK10K_DOMAIN

__all__ = ['MILK10kPipeline', 'MILK10K_DOMAIN']

# ================================

# config/__init__.py
"""
Configuration module for MILK10k pipeline
"""

from .settings import (
    MILK10K_DOMAIN,
    DATASET_PATH,
    GROUNDTRUTH_PATH,
    OUTPUT_PATH,
    SEGMENTATION_CONFIG,
    CLASSIFICATION_CONFIG,
    get_device,
    create_output_directories,
    validate_paths
)

__all__ = [
    'MILK10K_DOMAIN',
    'DATASET_PATH', 
    'GROUNDTRUTH_PATH',
    'OUTPUT_PATH',
    'SEGMENTATION_CONFIG',
    'CLASSIFICATION_CONFIG',
    'get_device',
    'create_output_directories',
    'validate_paths'
]

# ================================

# core/__init__.py
"""
Core pipeline components
"""

from .pipeline import MILK10kPipeline
from .domain import (
    MILK10K_DOMAIN,
    DomainManager,
    MedicalCondition,
    DomainConfiguration,
    ImageModalityType,
    DiseaseCategory,
    AnatomicalRegion
)

__all__ = [
    'MILK10kPipeline',
    'MILK10K_DOMAIN', 
    'DomainManager',
    'MedicalCondition',
    'DomainConfiguration',
    'ImageModalityType',
    'DiseaseCategory',
    'AnatomicalRegion'
]

# ================================

# preprocessing/__init__.py
"""
Image preprocessing and loading utilities
"""

from .image_loader import ImageLoader, validate_image_format, batch_load_images

__all__ = ['ImageLoader', 'validate_image_format', 'batch_load_images']

# ================================

# segmentation/__init__.py
"""
Image segmentation utilities using SAM2
"""

from .sam_segmenter import SAMSegmenter, validate_mask

__all__ = ['SAMSegmenter', 'validate_mask']

# ================================

# classification/__init__.py
"""
Image classification utilities using ConceptCLIP
"""

from .conceptclip_classifier import (
    ConceptCLIPClassifier,
    validate_text_prompts,
    create_medical_prompts,
    evaluate_classification_performance
)

__all__ = [
    'ConceptCLIPClassifier',
    'validate_text_prompts', 
    'create_medical_prompts',
    'evaluate_classification_performance'
]

# ================================

# utils/__init__.py
"""
Utility functions and helper classes
"""

from .auth import setup_huggingface_auth, verify_authentication
from .file_utils import (
    FileManager, 
    ConfigManager,
    save_processing_results,
    load_processing_results,
    create_backup,
    cleanup_temporary_files
)
from .visualization import Visualizer, save_segmentation_overlay, create_comparison_grid
from .conceptclip_analysis import ConceptCLIPAnalyzer

__all__ = [
    'setup_huggingface_auth',
    'verify_authentication',
    'FileManager',
    'ConfigManager', 
    'save_processing_results',
    'load_processing_results',
    'create_backup',
    'cleanup_temporary_files',
    'Visualizer',
    'save_segmentation_overlay',
    'create_comparison_grid',
    'ConceptCLIPAnalyzer'
]
