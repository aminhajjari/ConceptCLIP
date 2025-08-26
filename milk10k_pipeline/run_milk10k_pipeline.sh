#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8           # Increased for medical image processing
#SBATCH --mem=64G                   # Increased for SAM2 + ConceptCLIP + image processing
#SBATCH --gres=gpu:1                # Request 1 GPU for SAM2 and ConceptCLIP
#SBATCH --time=03:00:00             # 3 hours for full dataset processing
#SBATCH --mail-user=amminhajjari@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=milk10k_pipeline

# Change to your MILK10k project directory
cd /project/def-arashmoh/shahab33/XAI/milk10k_pipeline

# Load required modules
module purge
module load python
module load cuda
module load gcc/9.3.0              # Often needed for PyTorch compilation

# Activate your virtual environment
source /project/def-arashmoh/shahab33/XAI/milk10k_pipeline/venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/project/def-arashmoh/shahab33/.cache/huggingface
export TORCH_HOME=/project/def-arashmoh/shahab33/.cache/torch

echo "=== MILK10k Pipeline Started ==="
echo "Node: $SLURM_NODEID"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "PyTorch CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "=================================="

# Create organized output structure
echo "📁 Creating organized output structure..."
mkdir -p /project/def-arashmoh/shahab33/XAI/outputs/{01_segmented_images,02_classification_results,03_analysis_reports,04_visualizations,05_conceptclip_detailed,06_raw_data}

# Download ConceptCLIP if not cached (Naval downloads it automatically)
if [ ! -d "./models/conceptclip" ]; then
    echo "📥 Naval downloading ConceptCLIP model..."
    python utils/quick_download_conceptclip.py
    if [ $? -eq 0 ]; then
        echo "✅ ConceptCLIP download complete"
    else
        echo "❌ ConceptCLIP download failed"
        exit 1
    fi
else
    echo "✅ ConceptCLIP already cached"
fi

# Run the MILK10k pipeline with your XAI dataset paths
echo "🚀 Starting MILK10k pipeline..."
python main.py \
    --max-images 100 \
    --visualizations \
    --output-dir /project/def-arashmoh/shahab33/XAI/outputs \
    --dataset-path /project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/MILK10k_Training_Input \
    --groundtruth-path /project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/groundtruth.csv

echo "✅ MILK10k pipeline completed"

# Organize outputs into better structure
echo "🗂️  Organizing outputs..."

# Move segmented images
if [ -d "/project/def-arashmoh/shahab33/XAI/outputs/segmented_for_conceptclip" ]; then
    mv /project/def-arashmoh/shahab33/XAI/outputs/segmented_for_conceptclip/* /project/def-arashmoh/shahab33/XAI/outputs/01_segmented_images/
    echo "✅ Segmented images moved to 01_segmented_images/"
fi

# Move classification results
if [ -f "/project/def-arashmoh/shahab33/XAI/outputs/reports/detailed_results.csv" ]; then
    cp /project/def-arashmoh/shahab33/XAI/outputs/reports/detailed_results.csv /project/def-arashmoh/shahab33/XAI/outputs/02_classification_results/
    echo "✅ Classification results copied to 02_classification_results/"
fi

# Move analysis reports
if [ -d "/project/def-arashmoh/shahab33/XAI/outputs/reports" ]; then
    cp -r /project/def-arashmoh/shahab33/XAI/outputs/reports/* /project/def-arashmoh/shahab33/XAI/outputs/03_analysis_reports/
    echo "✅ Analysis reports copied to 03_analysis_reports/"
fi

# Move visualizations
if [ -d "/project/def-arashmoh/shahab33/XAI/outputs/visualizations" ]; then
    mv /project/def-arashmoh/shahab33/XAI/outputs/visualizations/* /project/def-arashmoh/shahab33/XAI/outputs/04_visualizations/
    echo "✅ Visualizations moved to 04_visualizations/"
fi

# Move ConceptCLIP detailed analysis
if [ -d "/project/def-arashmoh/shahab33/XAI/outputs/conceptclip_analysis" ]; then
    mv /project/def-arashmoh/shahab33/XAI/outputs/conceptclip_analysis/* /project/def-arashmoh/shahab33/XAI/outputs/05_conceptclip_detailed/
    echo "✅ ConceptCLIP analysis moved to 05_conceptclip_detailed/"
fi

# Create summary file
echo "📋 Creating output summary..."
cat > /project/def-arashmoh/shahab33/XAI/outputs/OUTPUT_SUMMARY.md << EOF
# MILK10k Pipeline Output Summary
Generated: $(date)
Job ID: $SLURM_JOB_ID

## 📁 Output Structure

### 01_segmented_images/
- Segmented images for each input image
- 5 types per image: colored_overlay, contour, cropped, masked_only, side_by_side

### 02_classification_results/
- detailed_results.csv: Main results with ConceptCLIP percentages
- All disease predictions and confidence scores

### 03_analysis_reports/
- processing_report.json: Overall processing statistics
- Detailed analysis reports

### 04_visualizations/
- summary_plots.png: Main summary charts
- per_class_performance.png: Disease-specific analysis
- Various analysis charts

### 05_conceptclip_detailed/
- Individual image analysis (JSON files)
- Comparison analysis (CSV files)
- Classification matrices
- Simplified results for easy analysis

### 06_raw_data/
- Raw processing logs and intermediate files

## 📊 Quick Stats
EOF

# Add stats to summary if files exist
if [ -f "/project/def-arashmoh/shahab33/XAI/outputs/03_analysis_reports/processing_report.json" ]; then
    python -c "
import json
try:
    with open('/project/def-arashmoh/shahab33/XAI/outputs/03_analysis_reports/processing_report.json') as f:
        report = json.load(f)
    print(f'Images processed: {report[\"dataset_info\"][\"total_images_found\"]}')
    print(f'Success rate: {report[\"dataset_info\"][\"processing_success_rate\"]:.1%}')
    if 'accuracy_metrics' in report:
        print(f'Overall accuracy: {report[\"accuracy_metrics\"][\"overall_accuracy\"]:.1%}')
except:
    print('Stats not available')
" >> /project/def-arashmoh/shahab33/XAI/outputs/OUTPUT_SUMMARY.md
fi

echo "✅ Output organization complete"
echo ""
echo "📊 Results saved in organized structure:"
echo "   📁 01_segmented_images/     - Segmented images"
echo "   📊 02_classification_results/ - Main results & percentages"
echo "   📋 03_analysis_reports/      - Processing reports"
echo "   📈 04_visualizations/        - Charts & plots"
echo "   🔬 05_conceptclip_detailed/  - Detailed ConceptCLIP analysis"
echo "   📄 OUTPUT_SUMMARY.md         - Complete summary"
echo ""
echo "🎉 Job completed successfully!"

# Show final directory structure
echo "=== FINAL OUTPUT STRUCTURE ==="
ls -la /project/def-arashmoh/shahab33/XAI/outputs/
echo "==============================="