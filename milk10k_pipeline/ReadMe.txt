# MILK10k Medical Image Processing Pipeline

A comprehensive pipeline for medical image segmentation and classification using SAM2 and ConceptCLIP on the MILK10k dataset.

## 🏗️ Project Structure

```
milk10k_pipeline/
├── main.py                                
├── utils/
│   ├── __init__.py                       
│   ├── auth.py                           
│   ├── quick_download_conceptclip.py     
│   ├── file_utils.py                     
│   └── visualization.py                  
├── models/                               # 🆕 Created after download
│   └── conceptclip/                      
├── config/
├── core/
├── preprocessing/
├── segmentation/
├── classification/
└── ...
```

## 🚀 Features

- **Multi-format Support**: DICOM, NIfTI, PNG, JPEG, TIFF
- **Advanced Segmentation**: SAM2 with adaptive strategies
- **Medical Classification**: ConceptCLIP for disease classification
- **Comprehensive Visualization**: Detailed plots and sample visualizations
- **Batch Processing**: Efficient processing of large datasets
- **Resume Capability**: Continue from interrupted runs
- **Flexible Configuration**: Customizable settings and parameters

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM
- 50GB+ storage for outputs

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/milk10k-pipeline.git
cd milk10k-pipeline
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install SAM2

```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### 5. Setup HuggingFace Authentication

Choose one of these methods:

**Option A: Environment Variable**
```bash
export HF_TOKEN=your_huggingface_token_here
```

**Option B: HuggingFace CLI**
```bash
huggingface-cli login
```

**Option C: Interactive Login**
The pipeline will prompt for login if not authenticated.

## ⚙️ Configuration

Edit `config/settings.py` to customize paths and parameters:

```python
# Dataset paths
DATASET_PATH = "/path/to/your/MILK10k_dataset"
GROUNDTRUTH_PATH = "/path/to/your/groundtruth.csv"
OUTPUT_PATH = "/path/to/your/outputs"

# Model configurations
SAM2_MODEL = "facebook/sam2-hiera-large"
CONCEPTCLIP_MODEL = "JerrryNie/ConceptCLIP"
```

## 🏃 Usage

### Basic Usage

Process entire dataset:
```bash
python main.py
```

### Testing Mode

Process only 5 images with visualizations:
```bash
python main.py --test-mode
```

### Custom Parameters

```bash
python main.py \
    --max-images 100 \
    --visualizations \
    --output-dir ./my_outputs \
    --dataset-path ./my_dataset
```

### Command Line Options

```
--max-images N          Process only first N images
--batch-size N          Process images in batches
--visualizations        Create sample visualizations
--test-mode            Quick test with 5 images
--output-dir PATH       Custom output directory
--dataset-path PATH     Custom dataset path
--groundtruth-path PATH Custom ground truth file
--resume               Resume from previous run
--help                 Show help message
```

## 📊 Output Structure

```
outputs/
├── segmented_for_conceptclip/      # Segmented images for classification
│   ├── image1/
│   │   ├── colored_overlay.png
│   │   ├── contour.png
│   │   ├── cropped.png
│   │   ├── masked_only.png
│   │   └── side_by_side.png
│   └── ...
├── reports/
│   ├── detailed_results.csv        # Detailed processing results
│   ├── processing_report.json      # Comprehensive report
│   └── ...
└── visualizations/
    ├── summary_plots.png           # Main summary plots
    ├── per_class_performance.png   # Per-class analysis
    └── sample_*.png               # Individual samples
```

## 🔍 Key Components

### 1. Image Preprocessing (`preprocessing/image_loader.py`)
- Supports DICOM, NIfTI, and standard formats
- CLAHE contrast enhancement
- Normalization and format conversion

### 2. Segmentation (`segmentation/sam_segmenter.py`)
- SAM2-based segmentation
- Adaptive ROI detection
- Multiple segmentation strategies
- Quality metrics calculation

### 3. Classification (`classification/conceptclip_classifier.py`)
- ConceptCLIP-based disease classification
- Ensemble prediction from multiple views
- Confidence-based filtering
- Medical domain prompts

### 4. Visualization (`utils/visualization.py`)
- Comprehensive summary plots
- Per-class performance analysis
- Sample visualizations
- Processing timelines

## 🎯 Medical Domain Configuration

The pipeline is configured for medical imaging with:

- **10 Disease Categories**: Normal, Abnormal, Inflammatory, Neoplastic, etc.
- **Medical Text Prompts**: Optimized for medical image classification
- **Adaptive Processing**: Handles various medical image formats
- **Quality Metrics**: Specialized metrics for medical image analysis

## 📈 Performance Monitoring

The pipeline provides detailed metrics:

- **Processing Success Rates**
- **Segmentation Quality Metrics**
- **Classification Confidence Scores**
- **Per-class Accuracy** (when ground truth available)
- **Error Analysis and Debugging Info**

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or use CPU mode
   - Close other GPU-intensive applications

2. **HuggingFace Authentication**
   - Ensure HF_TOKEN is set or login via CLI
   - Check token permissions for model access

3. **File Format Issues**
   - Verify DICOM files are valid
   - Check file permissions and paths

4. **SAM2 Installation**
   - Install from source if pip fails
   - Ensure compatible PyTorch version

### Debug Mode

Run with Python's debug flag for detailed error info:
```bash
python -u main.py --test-mode 2>&1 | tee debug.log
```

## 📝 Example Usage Scripts

### Process Subset for Quick Testing

```python
from core.pipeline import MILK10kPipeline

pipeline = MILK10kPipeline(
    dataset_path="./data/MILK10k",
    groundtruth_path="./data/groundtruth.csv", 
    output_path="./outputs"
)

# Process first 10 images
report = pipeline.process_dataset(max_images=10, create_sample_visualizations=True)
print(f"Accuracy: {report['accuracy_metrics']['overall_accuracy']:.2%}")
```

### Custom Configuration

```python
from config.settings import MILK10K_DOMAIN

# Modify domain configuration
MILK10K_DOMAIN.text_prompts = [
    'medical image showing healthy tissue',
    'medical image showing disease pathology',
    # Add your custom prompts
]

# Use custom domain
pipeline = MILK10kPipeline(...)
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **SAM2**: Meta's Segment Anything Model 2
- **ConceptCLIP**: Medical concept classification model
- **MILK10k Dataset**: Medical imaging dataset
- **HuggingFace**: Model hosting and transformers library

## 📧 Support

For questions or issues:
- Open a GitHub issue
- Email: your.email@domain.com
- Documentation: [Link to docs]

---

**Happy Medical Image Processing! 🏥🔬**