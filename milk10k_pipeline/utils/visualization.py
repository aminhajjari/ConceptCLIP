"""
Visualization utilities for MILK10k pipeline
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter
import cv2

class Visualizer:
    """Handles all visualization tasks for the pipeline"""
    
    def __init__(self, output_path: Path, plot_config: Dict[str, Any]):
        self.output_path = Path(output_path)
        self.plot_config = plot_config
        self.viz_dir = self.output_path / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        
        # Set plotting style
        plt.style.use(plot_config.get('style', 'default'))
        sns.set_palette(plot_config.get('color_palette', 'viridis'))
    
    def create_summary_plots(self, results_df: pd.DataFrame, report: Dict) -> Path:
        """Create comprehensive summary plots"""
        figsize = self.plot_config.get('figsize', (15, 12))
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Prediction distribution
        self._plot_prediction_distribution(axes[0, 0], report)
        
        # 2. Confidence distributions
        self._plot_confidence_distributions(axes[0, 1], results_df)
        
        # 3. Accuracy by confidence level
        self._plot_accuracy_by_confidence(axes[1, 0], results_df)
        
        # 4. Processing success rates
        self._plot_success_rates(axes[1, 1], report)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.viz_dir / "summary_plots.png"
        dpi = self.plot_config.get('dpi', 300)
        plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _plot_prediction_distribution(self, ax, report: Dict):
        """Plot distribution of predictions"""
        pred_counts = report.get('predictions', {}).get('distribution', {})
        
        if pred_counts:
            diseases = list(pred_counts.keys())
            counts = list(pred_counts.values())
            
            bars = ax.bar(diseases, counts)
            ax.set_title('Disease Prediction Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicted Disease')
            ax.set_ylabel('Number of Images')
            
            # Rotate labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{count}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No prediction data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Disease Prediction Distribution')
    
    def _plot_confidence_distributions(self, ax, results_df: pd.DataFrame):
        """Plot confidence score distributions"""
        if 'segmentation_confidence' in results_df.columns and 'prediction_confidence' in results_df.columns:
            seg_conf = results_df['segmentation_confidence'].dropna()
            pred_conf = results_df['prediction_confidence'].dropna()
            
            ax.hist([seg_conf, pred_conf], bins=30, alpha=0.7, 
                   label=['Segmentation', 'Classification'], density=True)
            ax.set_title('Confidence Score Distributions', fontsize=14, fontweight='bold')
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Density')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No confidence data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Confidence Score Distributions')
    
    def _plot_accuracy_by_confidence(self, ax, results_df: pd.DataFrame):
        """Plot accuracy as a function of confidence"""
        if 'ground_truth' in results_df.columns and 'prediction_confidence' in results_df.columns:
            results_with_gt = results_df.dropna(subset=['ground_truth', 'correct'])
            
            if len(results_with_gt) > 10:  # Need sufficient data
                # Create confidence bins
                conf_bins = np.linspace(0, 1, 11)
                accuracies = []
                bin_centers = []
                
                for i in range(len(conf_bins)-1):
                    mask = ((results_with_gt['prediction_confidence'] >= conf_bins[i]) & 
                           (results_with_gt['prediction_confidence'] < conf_bins[i+1]))
                    
                    if mask.sum() > 0:
                        acc = results_with_gt[mask]['correct'].mean()
                        accuracies.append(acc)
                        bin_centers.append((conf_bins[i] + conf_bins[i+1]) / 2)
                
                if accuracies:
                    ax.plot(bin_centers, accuracies, marker='o', linewidth=2, markersize=6)
                    ax.set_title('Accuracy vs Prediction Confidence', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Prediction Confidence')
                    ax.set_ylabel('Accuracy')
                    ax.set_ylim(0, 1)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'Insufficient data for analysis', 
                           ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'Insufficient ground truth data', 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No accuracy data available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('Accuracy vs Prediction Confidence')
    
    def _plot_success_rates(self, ax, report: Dict):
        """Plot processing success rates"""
        processing_stats = report.get('processing_stats', {})
        accuracy_metrics = report.get('accuracy_metrics', {})
        
        metrics = [
            ('Segmentation', processing_stats.get('segmentation_success_rate', 0)),
            ('Classification', processing_stats.get('classification_success_rate', 0)),
            ('Overall Accuracy', accuracy_metrics.get('overall_accuracy', 0))
        ]
        
        labels, values = zip(*metrics)
        
        bars = ax.bar(labels, values, color=['skyblue', 'lightgreen', 'salmon'])
        ax.set_title('Processing Success Rates', fontsize=14, fontweight='bold')
        ax.set_ylabel('Success Rate')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.2%}', ha='center', va='bottom')
    
    def create_detailed_analysis_plots(self, results_df: pd.DataFrame, report: Dict) -> List[Path]:
        """Create detailed analysis plots"""
        plot_paths = []
        
        # 1. Per-class performance
        if 'ground_truth' in results_df.columns:
            path = self._plot_per_class_performance(results_df)
            if path:
                plot_paths.append(path)
        
        # 2. Confidence correlation matrix
        path = self._plot_confidence_correlations(results_df)
        if path:
            plot_paths.append(path)
        
        # 3. File format analysis
        path = self._plot_format_analysis(report)
        if path:
            plot_paths.append(path)
        
        return plot_paths
    
    def _plot_per_class_performance(self, results_df: pd.DataFrame) -> Optional[Path]:
        """Plot per-class performance metrics"""
        try:
            results_with_gt = results_df.dropna(subset=['ground_truth'])
            
            if len(results_with_gt) == 0:
                return None
            
            # Calculate per-class metrics
            class_stats = results_with_gt.groupby('ground_truth').agg({
                'correct': ['count', 'sum', 'mean'],
                'prediction_confidence': ['mean', 'std']
            }).round(3)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Accuracy by class
            class_names = class_stats.index
            accuracies = class_stats['correct']['mean']
            
            bars1 = ax1.bar(class_names, accuracies)
            ax1.set_title('Accuracy by Disease Class', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0, 1)
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            
            # Add value labels
            for bar, acc in zip(bars1, accuracies):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{acc:.2%}', ha='center', va='bottom')
            
            # Sample counts by class
            sample_counts = class_stats['correct']['count']
            bars2 = ax2.bar(class_names, sample_counts)
            ax2.set_title('Sample Count by Disease Class', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Number of Samples')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            
            # Add value labels
            for bar, count in zip(bars2, sample_counts):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(count)}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            plot_path = self.viz_dir / "per_class_performance.png"
            plt.savefig(plot_path, dpi=self.plot_config.get('dpi', 300), bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            print(f"Error creating per-class performance plot: {e}")
            return None
    
    def _plot_confidence_correlations(self, results_df: pd.DataFrame) -> Optional[Path]:
        """Plot correlation matrix of confidence scores"""
        try:
            confidence_cols = ['segmentation_confidence', 'prediction_confidence']
            
            if all(col in results_df.columns for col in confidence_cols):
                corr_data = results_df[confidence_cols].corr()
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
                           square=True, cbar_kws={'label': 'Correlation'})
                plt.title('Confidence Score Correlations', fontsize=14, fontweight='bold')
                
                plot_path = self.viz_dir / "confidence_correlations.png"
                plt.savefig(plot_path, dpi=self.plot_config.get('dpi', 300), bbox_inches='tight')
                plt.close()
                
                return plot_path
            
        except Exception as e:
            print(f"Error creating confidence correlation plot: {e}")
        
        return None
    
    def _plot_format_analysis(self, report: Dict) -> Optional[Path]:
        """Plot file format analysis"""
        try:
            file_formats = report.get('dataset_info', {}).get('file_formats', {})
            
            if file_formats:
                plt.figure(figsize=(10, 6))
                
                formats = list(file_formats.keys())
                counts = list(file_formats.values())
                
                # Create pie chart
                plt.pie(counts, labels=formats, autopct='%1.1f%%', startangle=90)
                plt.title('Distribution of Image File Formats', fontsize=14, fontweight='bold')
                
                plot_path = self.viz_dir / "format_distribution.png"
                plt.savefig(plot_path, dpi=self.plot_config.get('dpi', 300), bbox_inches='tight')
                plt.close()
                
                return plot_path
            
        except Exception as e:
            print(f"Error creating format analysis plot: {e}")
        
        return None
    
    def create_sample_visualization(self, image_path: str, segmented_outputs: Dict[str, np.ndarray], 
                                   classification_result: Dict, output_name: str) -> Path:
        """Create visualization for a single sample"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Display different segmented outputs
        output_types = ['colored_overlay', 'contour', 'cropped', 'masked_only', 'side_by_side']
        
        for i, output_type in enumerate(output_types):
            if i < len(axes) - 1 and output_type in segmented_outputs:
                img = segmented_outputs[output_type]
                axes[i].imshow(img)
                axes[i].set_title(f'{output_type.replace("_", " ").title()}')
                axes[i].axis('off')
            else:
                axes[i].axis('off')
        
        # Classification results in the last subplot
        ensemble_probs = classification_result.get('ensemble_prediction', {})
        if ensemble_probs:
            diseases = list(ensemble_probs.keys())[:10]  # Top 10
            probs = [ensemble_probs[d] for d in diseases]
            
            axes[-1].barh(diseases, probs)
            axes[-1].set_title('Classification Probabilities')
            axes[-1].set_xlabel('Probability')
        
        plt.tight_layout()
        
        # Save sample visualization
        sample_path = self.viz_dir / f"sample_{output_name}.png"
        plt.savefig(sample_path, dpi=self.plot_config.get('dpi', 300), bbox_inches='tight')
        plt.close()
        
        return sample_path
    
    def create_processing_timeline(self, processing_stats: List[Dict]) -> Path:
        """Create timeline visualization of processing"""
        if not processing_stats:
            return None
        
        plt.figure(figsize=(12, 6))
        
        # Extract timing information (if available)
        times = [i for i in range(len(processing_stats))]
        success_rates = [stats.get('success_rate', 0) for stats in processing_stats]
        
        plt.plot(times, success_rates, marker='o', linewidth=2)
        plt.title('Processing Success Rate Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Batch Number')
        plt.ylabel('Success Rate')
        plt.grid(True, alpha=0.3)
        
        timeline_path = self.viz_dir / "processing_timeline.png"
        plt.savefig(timeline_path, dpi=self.plot_config.get('dpi', 300), bbox_inches='tight')
        plt.close()
        
        return timeline_path

# ==================== UTILITY FUNCTIONS ====================

def save_segmentation_overlay(image: np.ndarray, mask: np.ndarray, output_path: Path):
    """Save segmentation overlay image"""
    overlay = image.copy()
    overlay[mask == 1] = [255, 0, 0]  # Red overlay
    
    result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    cv2.imwrite(str(output_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

def create_comparison_grid(images: List[np.ndarray], titles: List[str], 
                          grid_size: tuple = (2, 3)) -> np.ndarray:
    """Create a grid of images for comparison"""
    rows, cols = grid_size
    
    if len(images) > rows * cols:
        images = images[:rows * cols]
    
    # Resize all images to the same size
    target_size = (300, 300)
    resized_images = []
    
    for img in images:
        resized = cv2.resize(img, target_size)
        resized_images.append(resized)
    
    # Create grid
    grid_rows = []
    for i in range(0, len(resized_images), cols):
        row_images = resized_images[i:i + cols]
        
        # Pad with black images if needed
        while len(row_images) < cols:
            row_images.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
        
        row = np.hstack(row_images)
        grid_rows.append(row)
    
    # Stack rows
    grid = np.vstack(grid_rows)
    
    return grid