"""
Image loading and preprocessing module for MILK10k pipeline
"""
import cv2
import numpy as np
import pydicom
import nibabel as nib
from pathlib import Path
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class ImageLoader:
    """Handles loading and preprocessing of various medical image formats"""
    
    def __init__(self, preprocessing_params: Dict[str, Any]):
        self.preprocessing_params = preprocessing_params
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Main entry point for loading images"""
        try:
            image_path = Path(image_path)
            ext = image_path.suffix.lower()
            
            if ext in ['.dcm', '.dicom']:
                return self._load_dicom(image_path)
            elif ext in ['.nii', '.nii.gz']:
                return self._load_nifti(image_path)
            else:
                return self._load_standard_image(image_path)
                
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    
    def _load_dicom(self, image_path: Path) -> np.ndarray:
        """Load DICOM images"""
        ds = pydicom.dcmread(image_path)
        image = ds.pixel_array.astype(np.float32)
        
        # Apply window/level if available
        if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
            window_center = ds.WindowCenter
            window_width = ds.WindowWidth
            
            # Handle multiple values
            if isinstance(window_center, (list, tuple)):
                window_center = window_center[0]
            if isinstance(window_width, (list, tuple)):
                window_width = window_width[0]
            
            # Apply windowing
            img_min = window_center - window_width // 2
            img_max = window_center + window_width // 2
            image = np.clip(image, img_min, img_max)
        
        # Normalize
        image = self._normalize_image(image)
        
        # Convert to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        return image
    
    def _load_nifti(self, image_path: Path) -> np.ndarray:
        """Load NIfTI images"""
        nii_img = nib.load(image_path)
        image = nii_img.get_fdata()
        
        # Handle 3D volumes by taking middle slice
        if len(image.shape) == 3:
            mid_slice = image.shape[2] // 2
            image = image[:, :, mid_slice]
        elif len(image.shape) == 4:
            # For 4D images, take middle slice of middle timepoint
            mid_time = image.shape[3] // 2
            mid_slice = image.shape[2] // 2
            image = image[:, :, mid_slice, mid_time]
        
        # Normalize
        image = self._normalize_image(image)
        
        # Convert to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        return image
    
    def _load_standard_image(self, image_path: Path) -> np.ndarray:
        """Load standard image formats (PNG, JPEG, TIFF, etc.)"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing
        if self.preprocessing_params.get('enhance_contrast', False):
            image = self._enhance_contrast(image)
        
        if self.preprocessing_params.get('normalize', False):
            image = self._normalize_image(image)
        
        return image
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-255 range"""
        image = image.astype(np.float32)
        
        # Handle edge case where all pixels are the same
        img_range = image.max() - image.min()
        if img_range == 0:
            return np.zeros_like(image, dtype=np.uint8)
        
        image = (image - image.min()) / img_range
        return (image * 255).astype(np.uint8)
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE"""
        clip_limit = self.preprocessing_params.get('clahe_clip_limit', 2.0)
        tile_grid_size = self.preprocessing_params.get('clahe_tile_grid_size', (8, 8))
        
        if len(image.shape) == 3:
            # For color images, apply CLAHE to L channel in LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # For grayscale images
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            return clahe.apply(image)
    
    def preprocess_for_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Additional preprocessing specifically for segmentation"""
        # Ensure image is in correct format for SAM2
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must be RGB format for segmentation")
        
        return image
    
    def get_image_info(self, image_path: Path) -> Dict[str, Any]:
        """Get basic information about the image file"""
        try:
            ext = image_path.suffix.lower()
            file_size = image_path.stat().st_size
            
            # Try to get image dimensions without fully loading
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                import PIL.Image
                with PIL.Image.open(image_path) as img:
                    width, height = img.size
                    mode = img.mode
            elif ext in ['.dcm', '.dicom']:
                ds = pydicom.dcmread(image_path, stop_before_pixels=True)
                height, width = ds.Rows, ds.Columns
                mode = "DICOM"
            elif ext in ['.nii', '.nii.gz']:
                nii_img = nib.load(image_path)
                shape = nii_img.shape
                height, width = shape[0], shape[1]
                mode = "NIfTI"
            else:
                height, width, mode = None, None, "Unknown"
            
            return {
                'file_path': str(image_path),
                'file_size_mb': file_size / (1024 * 1024),
                'format': ext,
                'dimensions': (width, height) if width and height else None,
                'mode': mode
            }
        
        except Exception as e:
            return {
                'file_path': str(image_path),
                'error': str(e)
            }

# ==================== UTILITY FUNCTIONS ====================

def validate_image_format(image_path: Path, supported_extensions: list) -> bool:
    """Validate if image format is supported"""
    ext = image_path.suffix.lower()
    return ext in supported_extensions

def batch_load_images(image_paths: list, preprocessing_params: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Load multiple images in batch"""
    loader = ImageLoader(preprocessing_params)
    results = {}
    
    for img_path in image_paths:
        image = loader.load_image(img_path)
        if image is not None:
            results[str(img_path)] = image
    
    return results