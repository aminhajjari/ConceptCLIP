"""
Medical domain definitions and configurations for MILK10k pipeline
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np

class ImageModalityType(Enum):
    """Medical image modality types"""
    CT = "CT"
    MRI = "MRI" 
    XRAY = "X-Ray"
    ULTRASOUND = "Ultrasound"
    MAMMOGRAPHY = "Mammography"
    PATHOLOGY = "Pathology"
    ENDOSCOPY = "Endoscopy"
    DERMATOLOGY = "Dermatology"
    OPHTHALMOLOGY = "Ophthalmology"
    GENERAL = "General"

class DiseaseCategory(Enum):
    """Medical disease categories"""
    NORMAL = "normal"
    ABNORMAL = "abnormal"
    INFLAMMATORY = "inflammatory"
    NEOPLASTIC = "neoplastic"
    DEGENERATIVE = "degenerative"
    INFECTIOUS = "infectious"
    VASCULAR = "vascular"
    METABOLIC = "metabolic"
    CONGENITAL = "congenital"
    TRAUMATIC = "traumatic"
    AUTOIMMUNE = "autoimmune"
    GENETIC = "genetic"

class AnatomicalRegion(Enum):
    """Anatomical regions for medical imaging"""
    HEAD_NECK = "head_neck"
    CHEST = "chest"
    ABDOMEN = "abdomen"
    PELVIS = "pelvis"
    SPINE = "spine"
    EXTREMITIES = "extremities"
    CARDIOVASCULAR = "cardiovascular"
    NEUROLOGICAL = "neurological"
    MUSCULOSKELETAL = "musculoskeletal"
    GASTROINTESTINAL = "gastrointestinal"

@dataclass
class MedicalCondition:
    """Represents a medical condition with metadata"""
    name: str
    category: DiseaseCategory
    description: str
    synonyms: List[str] = field(default_factory=list)
    anatomical_region: Optional[AnatomicalRegion] = None
    common_modalities: List[ImageModalityType] = field(default_factory=list)
    prevalence: Optional[str] = None
    urgency_level: str = "routine"  # routine, urgent, emergent
    
    def to_text_prompt(self, modality: Optional[str] = None) -> str:
        """Generate text prompt for this condition"""
        base_prompt = f"a medical image showing {self.name}"
        if modality:
            base_prompt = f"a {modality} medical image showing {self.name}"
        return base_prompt

@dataclass
class DomainConfiguration:
    """Configuration for a specific medical domain"""
    name: str
    description: str
    modality: ImageModalityType
    conditions: List[MedicalCondition]
    image_extensions: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.tiff', '.bmp'])
    preprocessing_params: Dict[str, Any] = field(default_factory=dict)
    segmentation_strategy: str = "adaptive"
    
    def get_text_prompts(self) -> List[str]:
        """Get all text prompts for this domain"""
        return [condition.to_text_prompt(self.modality.value) for condition in self.conditions]
    
    def get_label_mappings(self) -> Dict[str, str]:
        """Get label mappings for this domain"""
        mappings = {}
        for condition in self.conditions:
            # Primary name mapping
            key = condition.name.upper().replace(' ', '_').replace('-', '_')
            mappings[key] = condition.name
            
            # Synonyms mapping
            for synonym in condition.synonyms:
                syn_key = synonym.upper().replace(' ', '_').replace('-', '_')
                mappings[syn_key] = condition.name
        
        return mappings

# ==================== PREDEFINED MEDICAL CONDITIONS ====================

# Common medical conditions for general medical imaging
GENERAL_CONDITIONS = [
    MedicalCondition(
        name="normal tissue",
        category=DiseaseCategory.NORMAL,
        description="Healthy, normal tissue without pathological changes",
        synonyms=["healthy", "normal", "no abnormality"],
        urgency_level="routine"
    ),
    MedicalCondition(
        name="abnormal pathology", 
        category=DiseaseCategory.ABNORMAL,
        description="General abnormal findings requiring further evaluation",
        synonyms=["abnormal", "pathological", "disease"],
        urgency_level="urgent"
    ),
    MedicalCondition(
        name="inflammatory lesion",
        category=DiseaseCategory.INFLAMMATORY,
        description="Inflammatory process or lesion",
        synonyms=["inflammation", "inflammatory process", "inflammatory changes"],
        urgency_level="urgent"
    ),
    MedicalCondition(
        name="neoplastic lesion",
        category=DiseaseCategory.NEOPLASTIC,
        description="Neoplastic process, benign or malignant",
        synonyms=["tumor", "neoplasm", "mass", "growth"],
        urgency_level="emergent"
    ),
    MedicalCondition(
        name="degenerative changes",
        category=DiseaseCategory.DEGENERATIVE,
        description="Degenerative tissue changes",
        synonyms=["degeneration", "degenerative disease", "wear and tear"],
        urgency_level="routine"
    ),
    MedicalCondition(
        name="infectious disease",
        category=DiseaseCategory.INFECTIOUS,
        description="Infectious process or disease",
        synonyms=["infection", "infectious process", "septic"],
        urgency_level="urgent"
    ),
    MedicalCondition(
        name="vascular pathology",
        category=DiseaseCategory.VASCULAR,
        description="Vascular abnormalities or disease",
        synonyms=["vascular disease", "vascular abnormality", "vessel pathology"],
        urgency_level="urgent"
    ),
    MedicalCondition(
        name="metabolic disorder",
        category=DiseaseCategory.METABOLIC,
        description="Metabolic abnormalities or disorders",
        synonyms=["metabolic disease", "metabolic abnormality"],
        urgency_level="routine"
    ),
    MedicalCondition(
        name="congenital abnormality",
        category=DiseaseCategory.CONGENITAL,
        description="Congenital malformations or abnormalities",
        synonyms=["congenital defect", "birth defect", "congenital malformation"],
        urgency_level="routine"
    ),
    MedicalCondition(
        name="traumatic injury",
        category=DiseaseCategory.TRAUMATIC,
        description="Trauma-related injuries or changes",
        synonyms=["trauma", "injury", "traumatic changes"],
        urgency_level="emergent"
    )
]

# Chest/Pulmonary conditions
CHEST_CONDITIONS = [
    MedicalCondition(
        name="pneumonia",
        category=DiseaseCategory.INFECTIOUS,
        description="Lung infection and inflammation",
        synonyms=["lung infection", "pulmonary infection"],
        anatomical_region=AnatomicalRegion.CHEST,
        common_modalities=[ImageModalityType.CT, ImageModalityType.XRAY],
        urgency_level="urgent"
    ),
    MedicalCondition(
        name="pulmonary nodule",
        category=DiseaseCategory.NEOPLASTIC,
        description="Lung nodule or mass",
        synonyms=["lung nodule", "pulmonary mass"],
        anatomical_region=AnatomicalRegion.CHEST,
        common_modalities=[ImageModalityType.CT, ImageModalityType.XRAY],
        urgency_level="urgent"
    ),
    MedicalCondition(
        name="pneumothorax",
        category=DiseaseCategory.TRAUMATIC,
        description="Collapsed lung",
        synonyms=["collapsed lung", "air in pleural space"],
        anatomical_region=AnatomicalRegion.CHEST,
        common_modalities=[ImageModalityType.CT, ImageModalityType.XRAY],
        urgency_level="emergent"
    ),
    MedicalCondition(
        name="pleural effusion",
        category=DiseaseCategory.ABNORMAL,
        description="Fluid in pleural space",
        synonyms=["fluid in lung", "pleural fluid"],
        anatomical_region=AnatomicalRegion.CHEST,
        common_modalities=[ImageModalityType.CT, ImageModalityType.XRAY],
        urgency_level="urgent"
    )
]

# Neurological conditions
NEURO_CONDITIONS = [
    MedicalCondition(
        name="stroke",
        category=DiseaseCategory.VASCULAR,
        description="Cerebrovascular accident",
        synonyms=["cerebrovascular accident", "CVA", "brain infarct"],
        anatomical_region=AnatomicalRegion.NEUROLOGICAL,
        common_modalities=[ImageModalityType.CT, ImageModalityType.MRI],
        urgency_level="emergent"
    ),
    MedicalCondition(
        name="brain tumor",
        category=DiseaseCategory.NEOPLASTIC,
        description="Intracranial neoplasm",
        synonyms=["intracranial tumor", "brain mass", "cerebral tumor"],
        anatomical_region=AnatomicalRegion.NEUROLOGICAL,
        common_modalities=[ImageModalityType.CT, ImageModalityType.MRI],
        urgency_level="emergent"
    ),
    MedicalCondition(
        name="hemorrhage",
        category=DiseaseCategory.VASCULAR,
        description="Intracranial bleeding",
        synonyms=["brain bleed", "intracranial hemorrhage", "bleeding"],
        anatomical_region=AnatomicalRegion.NEUROLOGICAL,
        common_modalities=[ImageModalityType.CT, ImageModalityType.MRI],
        urgency_level="emergent"
    )
]

# Musculoskeletal conditions
MSK_CONDITIONS = [
    MedicalCondition(
        name="fracture",
        category=DiseaseCategory.TRAUMATIC,
        description="Bone fracture or break",
        synonyms=["bone break", "bone fracture", "broken bone"],
        anatomical_region=AnatomicalRegion.MUSCULOSKELETAL,
        common_modalities=[ImageModalityType.XRAY, ImageModalityType.CT],
        urgency_level="urgent"
    ),
    MedicalCondition(
        name="arthritis",
        category=DiseaseCategory.DEGENERATIVE,
        description="Joint inflammation and degeneration",
        synonyms=["joint degeneration", "osteoarthritis", "joint disease"],
        anatomical_region=AnatomicalRegion.MUSCULOSKELETAL,
        common_modalities=[ImageModalityType.XRAY, ImageModalityType.MRI],
        urgency_level="routine"
    )
]

# ==================== PREDEFINED DOMAIN CONFIGURATIONS ====================

# MILK10k General Domain (as originally configured)
MILK10K_GENERAL_DOMAIN = DomainConfiguration(
    name="milk10k_general",
    description="General medical imaging domain for MILK10k dataset",
    modality=ImageModalityType.GENERAL,
    conditions=GENERAL_CONDITIONS,
    image_extensions=['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.dcm', '.dicom', '.nii', '.nii.gz'],
    preprocessing_params={
        'normalize': True,
        'enhance_contrast': True,
        'clahe_clip_limit': 2.0,
        'clahe_tile_grid_size': (8, 8)
    },
    segmentation_strategy='adaptive'
)

# Chest X-ray Domain
CHEST_XRAY_DOMAIN = DomainConfiguration(
    name="chest_xray",
    description="Chest X-ray analysis domain",
    modality=ImageModalityType.XRAY,
    conditions=GENERAL_CONDITIONS + CHEST_CONDITIONS,
    image_extensions=['.jpg', '.jpeg', '.png', '.dcm', '.dicom'],
    preprocessing_params={
        'normalize': True,
        'enhance_contrast': True,
        'clahe_clip_limit': 3.0,
        'clahe_tile_grid_size': (16, 16)
    },
    segmentation_strategy='center_point'
)

# CT Scan Domain
CT_DOMAIN = DomainConfiguration(
    name="ct_scan",
    description="CT scan analysis domain",
    modality=ImageModalityType.CT,
    conditions=GENERAL_CONDITIONS + CHEST_CONDITIONS + NEURO_CONDITIONS,
    image_extensions=['.dcm', '.dicom', '.nii', '.nii.gz'],
    preprocessing_params={
        'normalize': True,
        'enhance_contrast': False,
        'windowing': True
    },
    segmentation_strategy='adaptive'
)

# MRI Domain
MRI_DOMAIN = DomainConfiguration(
    name="mri_scan",
    description="MRI scan analysis domain",
    modality=ImageModalityType.MRI,
    conditions=GENERAL_CONDITIONS + NEURO_CONDITIONS + MSK_CONDITIONS,
    image_extensions=['.dcm', '.dicom', '.nii', '.nii.gz'],
    preprocessing_params={
        'normalize': True,
        'enhance_contrast': False,
        'bias_correction': True
    },
    segmentation_strategy='multi_point'
)

# Pathology Domain
PATHOLOGY_DOMAIN = DomainConfiguration(
    name="pathology",
    description="Digital pathology analysis domain", 
    modality=ImageModalityType.PATHOLOGY,
    conditions=[
        MedicalCondition(
            name="malignant",
            category=DiseaseCategory.NEOPLASTIC,
            description="Malignant tissue",
            synonyms=["cancer", "malignancy", "malignant tumor"],
            urgency_level="emergent"
        ),
        MedicalCondition(
            name="benign",
            category=DiseaseCategory.NEOPLASTIC,
            description="Benign tissue",
            synonyms=["benign tumor", "non-malignant"],
            urgency_level="routine"
        ),
        MedicalCondition(
            name="normal tissue",
            category=DiseaseCategory.NORMAL,
            description="Normal histological tissue",
            synonyms=["healthy tissue", "normal histology"],
            urgency_level="routine"
        )
    ],
    image_extensions=['.jpg', '.jpeg', '.png', '.tiff', '.svs', '.ndpi'],
    preprocessing_params={
        'normalize': True,
        'enhance_contrast': True,
        'stain_normalization': True
    },
    segmentation_strategy='adaptive'
)

# ==================== DOMAIN UTILITIES ====================

class DomainManager:
    """Manages multiple medical domains"""
    
    def __init__(self):
        self.domains = {
            'milk10k': MILK10K_GENERAL_DOMAIN,
            'chest_xray': CHEST_XRAY_DOMAIN,
            'ct': CT_DOMAIN,
            'mri': MRI_DOMAIN,
            'pathology': PATHOLOGY_DOMAIN
        }
    
    def get_domain(self, name: str) -> Optional[DomainConfiguration]:
        """Get domain configuration by name"""
        return self.domains.get(name)
    
    def list_domains(self) -> List[str]:
        """List available domain names"""
        return list(self.domains.keys())
    
    def add_custom_domain(self, name: str, domain: DomainConfiguration):
        """Add custom domain configuration"""
        self.domains[name] = domain
    
    def get_conditions_by_category(self, domain_name: str, 
                                 category: DiseaseCategory) -> List[MedicalCondition]:
        """Get conditions by category for a specific domain"""
        domain = self.get_domain(domain_name)
        if not domain:
            return []
        
        return [condition for condition in domain.conditions 
                if condition.category == category]
    
    def get_conditions_by_urgency(self, domain_name: str, 
                                urgency: str) -> List[MedicalCondition]:
        """Get conditions by urgency level"""
        domain = self.get_domain(domain_name)
        if not domain:
            return []
        
        return [condition for condition in domain.conditions 
                if condition.urgency_level == urgency]
    
    def create_custom_condition_set(self, condition_names: List[str]) -> List[MedicalCondition]:
        """Create custom set of conditions from all domains"""
        all_conditions = []
        for domain in self.domains.values():
            all_conditions.extend(domain.conditions)
        
        # Remove duplicates based on name
        unique_conditions = {}
        for condition in all_conditions:
            unique_conditions[condition.name] = condition
        
        # Filter by requested names
        return [unique_conditions[name] for name in condition_names 
                if name in unique_conditions]

def create_domain_from_conditions(name: str, description: str, 
                                modality: ImageModalityType,
                                condition_names: List[str]) -> DomainConfiguration:
    """Create custom domain from list of condition names"""
    domain_manager = DomainManager()
    conditions = domain_manager.create_custom_condition_set(condition_names)
    
    return DomainConfiguration(
        name=name,
        description=description,
        modality=modality,
        conditions=conditions
    )

def get_condition_statistics(domain: DomainConfiguration) -> Dict[str, Any]:
    """Get statistics about conditions in a domain"""
    total_conditions = len(domain.conditions)
    
    # Count by category
    category_counts = {}
    for condition in domain.conditions:
        cat = condition.category.value
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    # Count by urgency
    urgency_counts = {}
    for condition in domain.conditions:
        urgency = condition.urgency_level
        urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1
    
    # Count by anatomical region
    region_counts = {}
    for condition in domain.conditions:
        if condition.anatomical_region:
            region = condition.anatomical_region.value
            region_counts[region] = region_counts.get(region, 0) + 1
    
    return {
        'total_conditions': total_conditions,
        'category_distribution': category_counts,
        'urgency_distribution': urgency_counts,
        'anatomical_distribution': region_counts,
        'modality': domain.modality.value,
        'supported_extensions': domain.image_extensions
    }

# ==================== LEGACY COMPATIBILITY ====================

# For backward compatibility with original MILK10k configuration
MILK10K_DOMAIN = MILK10K_GENERAL_DOMAIN

# Export commonly used items
__all__ = [
    'ImageModalityType',
    'DiseaseCategory', 
    'AnatomicalRegion',
    'MedicalCondition',
    'DomainConfiguration',
    'DomainManager',
    'MILK10K_DOMAIN',
    'MILK10K_GENERAL_DOMAIN',
    'CHEST_XRAY_DOMAIN',
    'CT_DOMAIN',
    'MRI_DOMAIN',
    'PATHOLOGY_DOMAIN',
    'create_domain_from_conditions',
    'get_condition_statistics'
]