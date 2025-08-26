"""
Main MILK10k Pipeline Orchestrator
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any, Optional
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
# Import custom modules
from config.settings import (
    SEGMENTATION_CONFIG, CLASSIFICATION_CONFIG,
    get_device, create_output_directories
)
from core.domain import MILK10K_DOMAIN
from utils.auth import setup_huggingface_auth
import os
from pathlib import Path
from preprocessing.image_loader import ImageLoader
from segmentation.sam_segmenter import SAMSegmenter
from classification.conceptclip_classifier import ConceptCLIPClassifier
from utils.visualization import Visualizer
from utils.conceptclip_analysis import ConceptCLIPAnalyzer

class MILK10kPipeline:
    """Main pipeline orchestrator for MILK10k processing"""
    
    def __init__(self, dataset_path: str, groundtruth_path: str, output_path: str, 
                 sam_model: str = "facebook/sam2-hiera-large", 
                 conceptclip_model: str = "JerrryNie/ConceptCLIP"):
        
        self.dataset_path = Path(dataset_path)
        self.groundtruth_path = groundtruth_path
        self.output_path = Path(output_path)
        self.domain = MILK10K_DOMAIN
        
        # Model configurations
        self.sam_model = sam_model
        self.conceptclip_model = conceptclip_model
        
        # Initialize device
        self.device = get_device()
        print(f"Initializing MILK10k pipeline on {self.device}")
        
        # Create output directories
        self.output_path = create_output_directories(self.output_path)
        
        # Initialize components
        self._initialize_components()
        
        # Load ground truth
        self.ground_truth = self._load_ground_truth()
    
    def _initialize_components(self):
    """Initialize all pipeline components"""
    print("Initializing pipeline components...")
    
    # Check if ConceptCLIP is cached
    cache_dir = Path("./models/conceptclip/transformers")
    if not cache_dir.exists():
        print("⚠️  ConceptCLIP not found in cache")
        print("💡 Run: python utils/quick_download_conceptclip.py")
        print("🔄 Falling back to online authentication...")
        if not setup_huggingface_auth():
            raise ValueError("Hugging Face authentication failed and no cached model found")
    else:
        print("✅ Using cached ConceptCLIP model")
        
        # Initialize image loader
        self.image_loader = ImageLoader(self.domain.preprocessing_params)
        
        # Initialize segmenter
        self.segmenter = SAMSegmenter(
            model_name=self.sam_model,
            device=self.device,
            segmentation_config=SEGMENTATION_CONFIG
        )
        
        # Initialize classifier
        self.classifier = ConceptCLIPClassifier(
            model_name=self.conceptclip_model,
            device=self.device,
            classification_config=CLASSIFICATION_CONFIG
        )
        
        # Initialize visualizer
        from config.settings import PLOT_CONFIG
        self.visualizer = Visualizer(self.output_path, PLOT_CONFIG)
        
        # Initialize ConceptCLIP analyzer
        self.conceptclip_analyzer = ConceptCLIPAnalyzer(self.output_path)
        
        print("All components initialized successfully")
    
    def _load_ground_truth(self) -> Optional[pd.DataFrame]:
        """Load ground truth annotations"""
        if os.path.exists(self.groundtruth_path):
            try:
                ground_truth = pd.read_csv(self.groundtruth_path)
                print(f"Loaded ground truth: {len(ground_truth)} samples")
                print(f"Ground truth columns: {list(ground_truth.columns)}")
                return ground_truth
            except Exception as e:
                print(f"Error loading ground truth: {e}")
                return None
        else:
            print(f"Ground truth file not found: {self.groundtruth_path}")
            return None
    
    def get_ground_truth_label(self, img_path: Path) -> Optional[str]:
        """Get ground truth label for image"""
        if self.ground_truth is None:
            return None
        
        img_name = img_path.stem
        
        # Try to find matching row in ground truth
        matching_rows = self.ground_truth[
            self.ground_truth.iloc[:, 0].astype(str).str.contains(img_name, na=False)
        ]
        
        if len(matching_rows) > 0:
            row = matching_rows.iloc[0]
            # Look for the label in subsequent columns
            for col in self.ground_truth.columns[1:]:
                if col in self.domain.label_mappings and row[col] == 1:
                    return self.domain.label_mappings[col]
            
            # If no specific column, check if there's a direct label column
            if 'label' in row:
                return str(row['label'])
        
        return None
    
    def process_single_image(self, image_path: Path) -> Dict[str, Any]:
        """Process a single image through the complete pipeline"""
        try:
            # Load and preprocess image
            image = self.image_loader.load_image(image_path)
            if image is None:
                return {'error': 'Failed to load image', 'image_path': str(image_path)}
            
            # Segment image
            mask, seg_confidence = self.segmenter.segment_image(image, self.domain.segmentation_strategy)
            
            # Create segmented outputs
            segmented_outputs = self.segmenter.create_segmented_outputs(image, mask)
            
            # Save segmented outputs
            img_name = image_path.stem
            conceptclip_dir = self.output_path / "segmented_for_conceptclip" / img_name
            conceptclip_dir.mkdir(exist_ok=True)
            
            for output_type, seg_image in segmented_outputs.items():
                if seg_image is not None and seg_image.size > 0:
                    output_path = conceptclip_dir / f"{output_type}.png"
                    import cv2
                    cv2.imwrite(str(output_path), cv2.cvtColor(seg_image, cv2.COLOR_RGB2BGR))
            
            # Classify using ConceptCLIP
            classification_result = self.classifier.classify_with_confidence_filtering(
                segmented_outputs, self.domain.text_prompts
            )
            
            # Get ground truth
            ground_truth = self.get_ground_truth_label(image_path)
            
            # Get prediction summary
            pred_summary = self.classifier.get_prediction_summary(classification_result)
            
            # Save detailed ConceptCLIP analysis
            segmented_paths = {output_type: str(conceptclip_dir / f"{output_type}.png") 
                             for output_type in segmented_outputs.keys()}
            
            conceptclip_file = self.conceptclip_analyzer.save_detailed_conceptclip_outputs(
                image_name=img_name,
                individual_predictions=classification_result.get('individual_predictions', {}),
                ensemble_prediction=classification_result.get('ensemble_prediction', {}),
                confidence_metrics=classification_result.get('confidence_metrics', {}),
                segmented_outputs_paths=segmented_paths
            )
            
            # Check accuracy if ground truth available
            correct = None
            if ground_truth and pred_summary.get('predicted_class'):
                correct = ground_truth == pred_summary['predicted_class']
            
            # Get segmentation quality metrics
            seg_metrics = self.segmenter.get_segmentation_quality_metrics(mask)
            
            return {
                'image_path': str(image_path),
                'image_name': img_name,
                'predicted_disease': pred_summary.get('predicted_class', 'unknown'),
                'prediction_confidence': pred_summary.get('confidence', 0.0),
                'segmentation_confidence': seg_confidence,
                'ground_truth': ground_truth,
                'correct': correct,
                'segmented_outputs_dir': str(conceptclip_dir),
                'classification_probabilities': classification_result.get('ensemble_prediction', {}),
                'segmentation_metrics': seg_metrics,
                'prediction_summary': pred_summary,
                'conceptclip_detailed_file': str(conceptclip_file),
                'processing_status': 'success'
            }
            
        except Exception as e:
            return {
                'image_path': str(image_path),
                'image_name': image_path.stem,
                'error': str(e),
                'processing_status': 'failed'
            }
    
    def process_dataset(self, max_images: Optional[int] = None, 
                       create_sample_visualizations: bool = False) -> Dict:
        """Process entire MILK10k dataset"""
        print("Starting MILK10k dataset processing...")
        
        # Find all images
        image_files = []
        for ext in self.domain.image_extensions:
            image_files.extend(self.dataset_path.rglob(f"*{ext}"))
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"Found {len(image_files)} images in dataset")
        
        results = []
        format_counter = Counter()
        correct_predictions = 0
        total_with_gt = 0
        processing_stats = []
        
        # Process images with progress bar
        for i, img_path in enumerate(tqdm(image_files, desc="Processing MILK10k images")):
            # Track file formats
            ext = img_path.suffix.lower()
            format_counter[ext] += 1
            
            # Process single image
            result = self.process_single_image(img_path)
            results.append(result)
            
            # Update statistics
            if result.get('processing_status') == 'success':
                if result.get('ground_truth'):
                    total_with_gt += 1
                    if result.get('correct'):
                        correct_predictions += 1
                
                # Progress indicator
                status = "✓" if result.get('correct') else ("✗" if result.get('ground_truth') else "-")
                pred_conf = result.get('prediction_confidence', 0)
                print(f"{status} {result['image_name']}: {result['predicted_disease']} ({pred_conf:.2%})")
                
                # Create sample visualization for first few successful cases
                if create_sample_visualizations and i < 5:
                    try:
                        # Load segmented outputs for visualization
                        segmented_outputs = self._load_segmented_outputs(result['segmented_outputs_dir'])
                        if segmented_outputs:
                            self.visualizer.create_sample_visualization(
                                result['image_path'],
                                segmented_outputs,
                                result.get('classification_probabilities', {}),
                                result['image_name']
                            )
                    except Exception as e:
                        print(f"Failed to create sample visualization for {result['image_name']}: {e}")
            else:
                print(f"✗ Failed to process {img_path}: {result.get('error', 'Unknown error')}")
            
            # Track processing statistics every 100 images
            if (i + 1) % 100 == 0:
                success_rate = sum(1 for r in results[-100:] if r.get('processing_status') == 'success') / 100
                processing_stats.append({
                    'batch': (i + 1) // 100,
                    'success_rate': success_rate,
                    'total_processed': i + 1
                })
        
        # Calculate final accuracy
        accuracy = correct_predictions / total_with_gt if total_with_gt > 0 else 0
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report(results, format_counter, accuracy, total_with_gt)
        
        # Save results and create visualizations
        self._save_results_and_visualizations(results, report, processing_stats)
        
        return report
    
    def _load_segmented_outputs(self, segmented_dir: str) -> Dict[str, np.ndarray]:
        """Load segmented outputs for visualization"""
        import cv2
        segmented_outputs = {}
        segmented_path = Path(segmented_dir)
        
        if segmented_path.exists():
            for output_file in segmented_path.glob("*.png"):
                output_type = output_file.stem
                image = cv2.imread(str(output_file))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    segmented_outputs[output_type] = image
        
        return segmented_outputs
    
    def _generate_comprehensive_report(self, results: List[Dict], format_counter: Counter, 
                                     accuracy: float, total_with_gt: int) -> Dict:
        """Generate comprehensive processing report"""
        
        # Filter successful results
        successful_results = [r for r in results if r.get('processing_status') == 'success']
        total_processed = len(results)
        successful_count = len(successful_results)
        
        # Basic statistics
        successful_segmentations = sum(1 for r in successful_results if r.get('segmentation_confidence', 0) > 0.5)
        successful_classifications = sum(1 for r in successful_results if r.get('prediction_confidence', 0) > 0.1)
        
        # Prediction distribution
        predictions = [r.get('predicted_disease', 'unknown') for r in successful_results]
        prediction_counts = Counter(predictions)
        
        # Confidence statistics
        seg_confidences = [r.get('segmentation_confidence', 0) for r in successful_results]
        pred_confidences = [r.get('prediction_confidence', 0) for r in successful_results]
        
        # Error analysis
        failed_results = [r for r in results if r.get('processing_status') == 'failed']
        error_types = Counter([r.get('error', 'Unknown error')[:50] for r in failed_results])
        
        report = {
            'dataset_info': {
                'total_images_found': total_processed,
                'successful_processing': successful_count,
                'failed_processing': len(failed_results),
                'processing_success_rate': successful_count / total_processed if total_processed > 0 else 0,
                'file_formats': dict(format_counter),
                'total_with_ground_truth': total_with_gt
            },
            'processing_stats': {
                'successful_segmentations': successful_segmentations,
                'successful_classifications': successful_classifications,
                'segmentation_success_rate': successful_segmentations / successful_count if successful_count > 0 else 0,
                'classification_success_rate': successful_classifications / successful_count if successful_count > 0 else 0
            },
            'accuracy_metrics': {
                'overall_accuracy': accuracy,
                'correct_predictions': sum(1 for r in successful_results if r.get('correct')),
                'total_evaluated': total_with_gt
            },
            'predictions': {
                'distribution': dict(prediction_counts),
                'most_common': prediction_counts.most_common(5)
            },
            'confidence_stats': {
                'segmentation': {
                    'mean': np.mean(seg_confidences) if seg_confidences else 0,
                    'std': np.std(seg_confidences) if seg_confidences else 0,
                    'min': np.min(seg_confidences) if seg_confidences else 0,
                    'max': np.max(seg_confidences) if seg_confidences else 0
                },
                'classification': {
                    'mean': np.mean(pred_confidences) if pred_confidences else 0,
                    'std': np.std(pred_confidences) if pred_confidences else 0,
                    'min': np.min(pred_confidences) if pred_confidences else 0,
                    'max': np.max(pred_confidences) if pred_confidences else 0
                }
            },
            'error_analysis': {
                'total_errors': len(failed_results),
                'error_types': dict(error_types),
                'most_common_errors': error_types.most_common(3)
            }
        }
        
        return report
    
    def _save_results_and_visualizations(self, results: List[Dict], report: Dict, 
                                       processing_stats: List[Dict]):
        """Save results and create visualizations"""
        
        # Save detailed results
        results_df = pd.DataFrame(results)
        results_path = self.output_path / "reports" / "detailed_results.csv"
        results_df.to_csv(results_path, index=False)
        
        # Save report
        report_path = self.output_path / "reports" / "processing_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create comprehensive ConceptCLIP analysis
        print("Creating comprehensive ConceptCLIP analysis...")
        conceptclip_analysis_files = self.conceptclip_analyzer.create_comprehensive_comparison_analysis(results_df)
        
        # Create visualizations
        print("Creating visualizations...")
        
        # Main summary plots
        summary_plot_path = self.visualizer.create_summary_plots(results_df, report)
        
        # Detailed analysis plots
        detailed_plot_paths = self.visualizer.create_detailed_analysis_plots(results_df, report)
        
        # Processing timeline
        if processing_stats:
            timeline_path = self.visualizer.create_processing_timeline(processing_stats)
        
        print(f"\nResults saved to: {self.output_path}")
        print(f"Segmented outputs for ConceptCLIP: {self.output_path / 'segmented_for_conceptclip'}")
        print(f"Detailed results: {results_path}")
        print(f"Processing report: {report_path}")
        print(f"Summary plots: {summary_plot_path}")
        
        if detailed_plot_paths:
            print("Detailed analysis plots:")
            for path in detailed_plot_paths:
                print(f"  - {path}")
        
        # Print ConceptCLIP analysis info
        print(f"\n🔬 CONCEPTCLIP COMPREHENSIVE ANALYSIS:")
        print(f"Detailed outputs: {self.output_path / 'conceptclip_analysis'}")
        print(f"Individual predictions: {self.output_path / 'conceptclip_analysis' / 'individual_outputs'}")
        print(f"Comparison analysis: {self.output_path / 'conceptclip_analysis' / 'comparisons'}")
        print(f"Matrices & reports: {self.output_path / 'conceptclip_analysis' / 'matrices'}")
        
        if conceptclip_analysis_files:
            print("Generated ConceptCLIP analyses:")
            for analysis_type, file_path in conceptclip_analysis_files.items():
                if file_path:
                    print(f"  ✓ {analysis_type}: {file_path.name}")
        
        print(f"ConceptCLIP summary: {self.output_path / 'conceptclip_analysis' / 'analysis_summary.json'}")
    
    def process_batch(self, image_paths: List[Path], batch_size: int = 10) -> List[Dict]:
        """Process a batch of images"""
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            batch_results = []
            
            print(f"Processing batch {i // batch_size + 1}/{(len(image_paths) + batch_size - 1) // batch_size}")
            
            for img_path in tqdm(batch, desc="Batch progress"):
                result = self.process_single_image(img_path)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline components and configuration"""
        return {
            'pipeline_config': {
                'device': self.device,
                'sam_model': self.sam_model,
                'conceptclip_model': self.conceptclip_model,
                'domain': self.domain.name,
                'segmentation_strategy': self.domain.segmentation_strategy
            },
            'dataset_info': {
                'dataset_path': str(self.dataset_path),
                'ground_truth_available': self.ground_truth is not None,
                'supported_formats': self.domain.image_extensions
            },
            'output_structure': {
                'base_path': str(self.output_path),
                'segmented_outputs': str(self.output_path / "segmented_for_conceptclip"),
                'reports': str(self.output_path / "reports"),
                'visualizations': str(self.output_path / "visualizations")
            }
        }