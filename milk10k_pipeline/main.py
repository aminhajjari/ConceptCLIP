#!/usr/bin/env python3
"""
MILK10k Medical Image Segmentation and Classification Pipeline
Main execution script

Usage:
    python main.py [options]
    
Options:
    --max-images N          Process only first N images (for testing)
    --batch-size N          Process images in batches of size N
    --visualizations        Create sample visualizations
    --config CONFIG_FILE    Use custom configuration file
    --output-dir PATH       Custom output directory
    --help                  Show this help message
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import pipeline components
from config.settings import (
    DATASET_PATH, GROUNDTRUTH_PATH, OUTPUT_PATH, 
    SAM2_MODEL, CONCEPTCLIP_MODEL, validate_paths
)
from core.pipeline import MILK10kPipeline
from utils.file_utils import ConfigManager

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="MILK10k Medical Image Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--max-images', 
        type=int, 
        default=None,
        help='Process only first N images (useful for testing)'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=1,
        help='Process images in batches (default: 1)'
    )
    
    parser.add_argument(
        '--visualizations', 
        action='store_true',
        help='Create sample visualizations for first few images'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default=None,
        help='Path to custom configuration file'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default=OUTPUT_PATH,
        help='Custom output directory path'
    )
    
    parser.add_argument(
        '--dataset-path', 
        type=str, 
        default=DATASET_PATH,
        help='Path to MILK10k dataset'
    )
    
    parser.add_argument(
        '--groundtruth-path', 
        type=str, 
        default=GROUNDTRUTH_PATH,
        help='Path to ground truth CSV file'
    )
    
    parser.add_argument(
        '--sam-model', 
        type=str, 
        default=SAM2_MODEL,
        help='SAM2 model name/path'
    )
    
    parser.add_argument(
        '--conceptclip-model', 
        type=str, 
        default=CONCEPTCLIP_MODEL,
        help='ConceptCLIP model name/path'
    )
    
    parser.add_argument(
        '--test-mode', 
        action='store_true',
        help='Run in test mode (process only 5 images with visualizations)'
    )
    
    parser.add_argument(
        '--resume', 
        action='store_true',
        help='Resume processing from previous run (if results exist)'
    )
    
    return parser.parse_args()

def load_custom_config(config_path: str):
    """Load custom configuration from file"""
    try:
        config_manager = ConfigManager(Path(config_path).parent)
        config_name = Path(config_path).stem
        return config_manager.load_config(config_name)
    except Exception as e:
        print(f"Warning: Failed to load custom config {config_path}: {e}")
        return None

def validate_environment():
    """Validate environment and dependencies"""
    print("Validating environment...")
    
    # Check paths
    try:
        validate_paths()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("⚠ CUDA not available, will use CPU (slower)")
    except ImportError:
        print("⚠ PyTorch not found")
        return False
    
    # Check required libraries
    required_libs = [
        'transformers', 'huggingface_hub', 'sam2', 
        'cv2', 'PIL', 'pandas', 'numpy', 'matplotlib'
    ]
    
    missing_libs = []
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing_libs.append(lib)
    
    if missing_libs:
        print(f"Error: Missing required libraries: {missing_libs}")
        return False
    
    # Check ConceptCLIP cache
    from pathlib import Path  # Add this import
    cache_dir = Path("./models/conceptclip/transformers")
    if cache_dir.exists():
        print("✅ ConceptCLIP cached model found")
    else:
        print("⚠️  ConceptCLIP not cached - will need authentication")
        print("💡 To cache ConceptCLIP: python utils/quick_download_conceptclip.py")
    
    print("✓ Environment validation passed")
    return True

def print_pipeline_summary(pipeline: MILK10kPipeline):
    """Print pipeline configuration summary"""
    summary = pipeline.get_processing_summary()
    
    print("\n" + "="*60)
    print("MILK10K PIPELINE CONFIGURATION")
    print("="*60)
    
    print(f"Device: {summary['pipeline_config']['device']}")
    print(f"SAM Model: {summary['pipeline_config']['sam_model']}")
    print(f"ConceptCLIP Model: {summary['pipeline_config']['conceptclip_model']}")
    print(f"Domain: {summary['pipeline_config']['domain']}")
    print(f"Segmentation Strategy: {summary['pipeline_config']['segmentation_strategy']}")
    
    print(f"\nDataset Path: {summary['dataset_info']['dataset_path']}")
    print(f"Ground Truth Available: {summary['dataset_info']['ground_truth_available']}")
    print(f"Supported Formats: {', '.join(summary['dataset_info']['supported_formats'])}")
    
    print(f"\nOutput Directory: {summary['output_structure']['base_path']}")
    print("="*60)

def main():
    """Main execution function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Handle test mode
    if args.test_mode:
        args.max_images = 5
        args.visualizations = True
        print("Running in TEST MODE: Processing only 5 images with visualizations")
    
    # Load custom configuration if provided
    custom_config = None
    if args.config:
        custom_config = load_custom_config(args.config)
        if custom_config:
            print(f"Loaded custom configuration from {args.config}")
    
    # Validate environment
    if not validate_environment():
        print("Environment validation failed. Please install required dependencies.")
        sys.exit(1)
    
    try:
        # Initialize pipeline
        print("Initializing MILK10k pipeline...")
        
        pipeline = MILK10kPipeline(
            dataset_path=args.dataset_path,
            groundtruth_path=args.groundtruth_path,
            output_path=args.output_dir,
            sam_model=args.sam_model,
            conceptclip_model=args.conceptclip_model
        )
        
        # Print configuration summary
        print_pipeline_summary(pipeline)
        
        # Check for resume option
        if args.resume:
            from utils.file_utils import load_processing_results
            existing_results = load_processing_results(Path(args.output_dir))
            if existing_results:
                print(f"Found {len(existing_results)} existing results. Resuming processing...")
                # Implementation for resuming would go here
                # For now, we'll just proceed with full processing
        
        # Process dataset
        print(f"\nStarting dataset processing...")
        if args.max_images:
            print(f"Processing maximum {args.max_images} images")
        
        report = pipeline.process_dataset(
            max_images=args.max_images,
            create_sample_visualizations=args.visualizations
        )
        
        # Print final summary
        print("\n" + "="*60)
        print("PROCESSING COMPLETE - SUMMARY")
        print("="*60)
        
        dataset_info = report['dataset_info']
        processing_stats = report['processing_stats']
        accuracy_metrics = report['accuracy_metrics']
        
        print(f"Total Images Found: {dataset_info['total_images_found']}")
        print(f"Successfully Processed: {dataset_info['successful_processing']}")
        print(f"Processing Success Rate: {dataset_info['processing_success_rate']:.2%}")
        
        if dataset_info.get('failed_processing', 0) > 0:
            print(f"Failed Processing: {dataset_info['failed_processing']}")
        
        print(f"\nSegmentation Success Rate: {processing_stats['segmentation_success_rate']:.2%}")
        print(f"Classification Success Rate: {processing_stats['classification_success_rate']:.2%}")
        
        if accuracy_metrics['total_evaluated'] > 0:
            print(f"\nOverall Accuracy: {accuracy_metrics['overall_accuracy']:.2%}")
            print(f"Correct Predictions: {accuracy_metrics['correct_predictions']}/{accuracy_metrics['total_evaluated']}")
        
        # Show top predictions
        top_predictions = report.get('predictions', {}).get('most_common', [])
        if top_predictions:
            print(f"\nTop Predicted Diseases:")
            for disease, count in top_predictions:
                print(f"  {disease}: {count} images")
        
        # Output locations
        print(f"\nResults saved to: {args.output_dir}")
        print("Key output directories:")
        print(f"  • Segmented outputs: {Path(args.output_dir) / 'segmented_for_conceptclip'}")
        print(f"  • Detailed reports: {Path(args.output_dir) / 'reports'}")
        print(f"  • Visualizations: {Path(args.output_dir) / 'visualizations'}")
        
        print("\n✓ Pipeline completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
        print("Partial results may be available in the output directory.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()