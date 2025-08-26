"""
Comprehensive ConceptCLIP analysis and comparison utilities
"""
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class ConceptCLIPAnalyzer:
    """Comprehensive analysis of ConceptCLIP outputs"""
    
    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.analysis_dir = self.output_path / "conceptclip_analysis"
        self.analysis_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for detailed analysis
        (self.analysis_dir / "individual_outputs").mkdir(exist_ok=True)
        (self.analysis_dir / "comparisons").mkdir(exist_ok=True)
        (self.analysis_dir / "matrices").mkdir(exist_ok=True)
        (self.analysis_dir / "rankings").mkdir(exist_ok=True)
    
    def save_detailed_conceptclip_outputs(self, image_name: str, 
                                        individual_predictions: Dict[str, Dict[str, float]],
                                        ensemble_prediction: Dict[str, float],
                                        confidence_metrics: Dict[str, Any],
                                        segmented_outputs_paths: Dict[str, str]) -> Path:
        """Save comprehensive ConceptCLIP outputs for single image"""
        
        # Create detailed output structure
        detailed_output = {
            'image_name': image_name,
            'timestamp': pd.Timestamp.now().isoformat(),
            'segmented_image_paths': segmented_outputs_paths,
            
            # Individual ConceptCLIP outputs for each segmentation type
            'individual_predictions': individual_predictions,
            
            # Ensemble results
            'ensemble_prediction': ensemble_prediction,
            'predicted_class': max(ensemble_prediction.items(), key=lambda x: x[1])[0],
            'prediction_confidence': max(ensemble_prediction.values()),
            
            # Confidence and consistency metrics
            'confidence_metrics': confidence_metrics,
            
            # Ranking analysis
            'prediction_rankings': self._create_prediction_rankings(individual_predictions, ensemble_prediction),
            
            # Consistency analysis
            'consistency_analysis': self._analyze_prediction_consistency(individual_predictions),
            
            # Disagreement analysis
            'disagreement_analysis': self._analyze_segmentation_disagreements(individual_predictions)
        }
        
        # Save detailed JSON
        output_file = self.analysis_dir / "individual_outputs" / f"{image_name}_conceptclip_detailed.json"
        with open(output_file, 'w') as f:
            json.dump(detailed_output, f, indent=2, default=str)
        
        # Save simplified CSV for easy analysis
        self._save_simplified_csv(image_name, detailed_output)
        
        return output_file
    
    def _create_prediction_rankings(self, individual_preds: Dict, ensemble_pred: Dict) -> Dict:
        """Create ranking analysis for predictions"""
        rankings = {
            'ensemble_ranking': sorted(ensemble_pred.items(), key=lambda x: x[1], reverse=True)
        }
        
        # Individual rankings for each segmentation type
        for seg_type, predictions in individual_preds.items():
            rankings[f'{seg_type}_ranking'] = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        # Top-3 consistency check
        ensemble_top3 = [item[0] for item in rankings['ensemble_ranking'][:3]]
        rankings['top3_consistency'] = {}
        
        for seg_type in individual_preds.keys():
            seg_top3 = [item[0] for item in rankings[f'{seg_type}_ranking'][:3]]
            overlap = len(set(ensemble_top3) & set(seg_top3))
            rankings['top3_consistency'][seg_type] = overlap / 3.0
        
        return rankings
    
    def _analyze_prediction_consistency(self, individual_preds: Dict) -> Dict:
        """Analyze consistency across segmentation types"""
        if not individual_preds:
            return {}
        
        # Get all diseases
        diseases = list(next(iter(individual_preds.values())).keys())
        
        consistency_metrics = {}
        
        for disease in diseases:
            disease_preds = [preds.get(disease, 0) for preds in individual_preds.values()]
            
            consistency_metrics[disease] = {
                'mean_confidence': np.mean(disease_preds),
                'std_confidence': np.std(disease_preds),
                'min_confidence': np.min(disease_preds),
                'max_confidence': np.max(disease_preds),
                'coefficient_of_variation': np.std(disease_preds) / (np.mean(disease_preds) + 1e-10)
            }
        
        # Overall consistency score
        all_cvs = [metrics['coefficient_of_variation'] for metrics in consistency_metrics.values()]
        consistency_metrics['overall_consistency'] = 1.0 - np.mean(all_cvs)  # Higher is more consistent
        
        return consistency_metrics
    
    def _analyze_segmentation_disagreements(self, individual_preds: Dict) -> Dict:
        """Analyze where different segmentation types disagree most"""
        if len(individual_preds) < 2:
            return {}
        
        seg_types = list(individual_preds.keys())
        disagreements = {}
        
        for i, seg_type1 in enumerate(seg_types):
            for seg_type2 in seg_types[i+1:]:
                preds1 = individual_preds[seg_type1]
                preds2 = individual_preds[seg_type2]
                
                # Calculate differences for each disease
                differences = {}
                for disease in preds1.keys():
                    diff = abs(preds1[disease] - preds2[disease])
                    differences[disease] = diff
                
                # Top disagreements
                top_disagreements = sorted(differences.items(), key=lambda x: x[1], reverse=True)[:5]
                
                disagreements[f'{seg_type1}_vs_{seg_type2}'] = {
                    'max_disagreement': max(differences.values()),
                    'mean_disagreement': np.mean(list(differences.values())),
                    'top_disagreements': top_disagreements
                }
        
        return disagreements
    
    def _save_simplified_csv(self, image_name: str, detailed_output: Dict):
        """Save simplified CSV row for batch analysis"""
        csv_file = self.analysis_dir / "conceptclip_simplified_results.csv"
        
        # Create simplified row
        row = {
            'image_name': image_name,
            'predicted_class': detailed_output['predicted_class'],
            'prediction_confidence': detailed_output['prediction_confidence'],
            'overall_consistency': detailed_output['consistency_analysis'].get('overall_consistency', 0)
        }
        
        # Add top-3 predictions with confidences
        ensemble_ranking = detailed_output['prediction_rankings']['ensemble_ranking']
        for i, (disease, conf) in enumerate(ensemble_ranking[:3]):
            row[f'top_{i+1}_prediction'] = disease
            row[f'top_{i+1}_confidence'] = conf
        
        # Add individual segmentation type results
        for seg_type, predictions in detailed_output['individual_predictions'].items():
            top_pred = max(predictions.items(), key=lambda x: x[1])
            row[f'{seg_type}_prediction'] = top_pred[0]
            row[f'{seg_type}_confidence'] = top_pred[1]
        
        # Save to CSV
        df_row = pd.DataFrame([row])
        if csv_file.exists():
            df_existing = pd.read_csv(csv_file)
            df_combined = pd.concat([df_existing, df_row], ignore_index=True)
            df_combined.to_csv(csv_file, index=False)
        else:
            df_row.to_csv(csv_file, index=False)
    
    def create_comprehensive_comparison_analysis(self, results_df: pd.DataFrame) -> Dict[str, Path]:
        """Create comprehensive comparison analysis across all images"""
        analysis_files = {}
        
        # 1. Segmentation Type Performance Comparison
        seg_comparison_file = self._analyze_segmentation_type_performance(results_df)
        analysis_files['segmentation_comparison'] = seg_comparison_file
        
        # 2. Disease Prediction Confidence Analysis
        disease_analysis_file = self._analyze_disease_prediction_patterns(results_df)
        analysis_files['disease_analysis'] = disease_analysis_file
        
        # 3. Consistency vs Accuracy Analysis
        consistency_file = self._analyze_consistency_vs_accuracy(results_df)
        analysis_files['consistency_analysis'] = consistency_file
        
        # 4. ConceptCLIP Confusion Matrix
        confusion_file = self._create_conceptclip_confusion_matrix(results_df)
        analysis_files['confusion_matrix'] = confusion_file
        
        # 5. Ensemble vs Individual Performance
        ensemble_file = self._analyze_ensemble_performance(results_df)
        analysis_files['ensemble_analysis'] = ensemble_file
        
        return analysis_files
    
    def _analyze_segmentation_type_performance(self, results_df: pd.DataFrame) -> Path:
        """Analyze which segmentation types perform best"""
        
        # Load detailed ConceptCLIP results
        simplified_results = self.analysis_dir / "conceptclip_simplified_results.csv"
        if not simplified_results.exists():
            return None
        
        concept_df = pd.read_csv(simplified_results)
        
        # Identify segmentation type columns
        seg_type_cols = [col for col in concept_df.columns if col.endswith('_prediction')]
        
        if not seg_type_cols:
            return None
        
        # Performance analysis
        performance_data = []
        
        for seg_col in seg_type_cols:
            seg_type = seg_col.replace('_prediction', '')
            conf_col = f'{seg_type}_confidence'
            
            if conf_col in concept_df.columns:
                performance_data.append({
                    'segmentation_type': seg_type,
                    'mean_confidence': concept_df[conf_col].mean(),
                    'std_confidence': concept_df[conf_col].std(),
                    'min_confidence': concept_df[conf_col].min(),
                    'max_confidence': concept_df[conf_col].max(),
                    'above_threshold_count': (concept_df[conf_col] > 0.5).sum(),
                    'high_confidence_count': (concept_df[conf_col] > 0.8).sum()
                })
        
        perf_df = pd.DataFrame(performance_data)
        
        # Save analysis
        output_file = self.analysis_dir / "comparisons" / "segmentation_type_performance.csv"
        perf_df.to_csv(output_file, index=False)
        
        # Create visualization
        self._visualize_segmentation_performance(perf_df)
        
        return output_file
    
    def _analyze_disease_prediction_patterns(self, results_df: pd.DataFrame) -> Path:
        """Analyze disease prediction patterns across all images"""
        
        # Collect all disease predictions
        disease_stats = {}
        
        # Load individual ConceptCLIP files for detailed analysis
        individual_files = list((self.analysis_dir / "individual_outputs").glob("*_conceptclip_detailed.json"))
        
        all_diseases = set()
        prediction_data = []
        
        for file_path in individual_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            ensemble_pred = data.get('ensemble_prediction', {})
            all_diseases.update(ensemble_pred.keys())
            
            # Collect prediction data
            for disease, confidence in ensemble_pred.items():
                prediction_data.append({
                    'image': data['image_name'],
                    'disease': disease,
                    'confidence': confidence,
                    'is_predicted': disease == data['predicted_class']
                })
        
        pred_df = pd.DataFrame(prediction_data)
        
        # Analyze patterns
        disease_analysis = []
        for disease in all_diseases:
            disease_data = pred_df[pred_df['disease'] == disease]
            
            disease_analysis.append({
                'disease': disease,
                'times_predicted': (disease_data['is_predicted']).sum(),
                'mean_confidence': disease_data['confidence'].mean(),
                'std_confidence': disease_data['confidence'].std(),
                'max_confidence': disease_data['confidence'].max(),
                'high_confidence_instances': (disease_data['confidence'] > 0.8).sum(),
                'prediction_rate': (disease_data['is_predicted']).sum() / len(individual_files)
            })
        
        analysis_df = pd.DataFrame(disease_analysis)
        analysis_df = analysis_df.sort_values('times_predicted', ascending=False)
        
        # Save analysis
        output_file = self.analysis_dir / "comparisons" / "disease_prediction_patterns.csv"
        analysis_df.to_csv(output_file, index=False)
        
        # Create visualization
        self._visualize_disease_patterns(analysis_df)
        
        return output_file
    
    def _analyze_consistency_vs_accuracy(self, results_df: pd.DataFrame) -> Path:
        """Analyze relationship between prediction consistency and accuracy"""
        
        if 'ground_truth' not in results_df.columns:
            return None
        
        # Load simplified results with consistency metrics
        simplified_file = self.analysis_dir / "conceptclip_simplified_results.csv"
        if not simplified_file.exists():
            return None
        
        concept_df = pd.read_csv(simplified_file)
        
        # Merge with ground truth
        merged_df = concept_df.merge(
            results_df[['image_name', 'ground_truth', 'correct']], 
            on='image_name', 
            how='inner'
        )
        
        # Consistency vs accuracy analysis
        consistency_bins = pd.cut(merged_df['overall_consistency'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        consistency_analysis = merged_df.groupby(consistency_bins).agg({
            'correct': ['count', 'sum', 'mean'],
            'prediction_confidence': ['mean', 'std'],
            'overall_consistency': 'mean'
        }).round(3)
        
        # Save analysis
        output_file = self.analysis_dir / "comparisons" / "consistency_vs_accuracy.csv"
        consistency_analysis.to_csv(output_file)
        
        # Create visualization
        self._visualize_consistency_accuracy(merged_df)
        
        return output_file
    
    def _create_conceptclip_confusion_matrix(self, results_df: pd.DataFrame) -> Path:
        """Create detailed confusion matrix for ConceptCLIP predictions"""
        
        if 'ground_truth' not in results_df.columns:
            return None
        
        # Filter rows with ground truth
        gt_df = results_df.dropna(subset=['ground_truth', 'predicted_disease'])
        
        if len(gt_df) == 0:
            return None
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        
        confusion_mat = confusion_matrix(gt_df['ground_truth'], gt_df['predicted_disease'])
        
        # Get unique labels
        labels = sorted(set(gt_df['ground_truth'].unique()) | set(gt_df['predicted_disease'].unique()))
        
        # Create detailed confusion matrix with percentages
        confusion_df = pd.DataFrame(confusion_mat, index=labels, columns=labels)
        
        # Calculate percentages
        confusion_pct = confusion_df.div(confusion_df.sum(axis=1), axis=0) * 100
        
        # Save matrices
        output_dir = self.analysis_dir / "matrices"
        confusion_df.to_csv(output_dir / "confusion_matrix_counts.csv")
        confusion_pct.to_csv(output_dir / "confusion_matrix_percentages.csv")
        
        # Classification report
        class_report = classification_report(
            gt_df['ground_truth'], 
            gt_df['predicted_disease'], 
            output_dict=True
        )
        
        class_report_df = pd.DataFrame(class_report).transpose()
        class_report_df.to_csv(output_dir / "classification_report.csv")
        
        # Create visualization
        self._visualize_confusion_matrix(confusion_pct, labels)
        
        return output_dir / "confusion_matrix_counts.csv"
    
    def _analyze_ensemble_performance(self, results_df: pd.DataFrame) -> Path:
        """Compare ensemble vs individual segmentation performance"""
        
        # This would require individual segmentation type predictions
        # For now, create analysis based on available data
        
        ensemble_analysis = {
            'total_images': len(results_df),
            'successful_predictions': (results_df['prediction_confidence'] > 0.1).sum(),
            'high_confidence_predictions': (results_df['prediction_confidence'] > 0.8).sum(),
            'mean_ensemble_confidence': results_df['prediction_confidence'].mean(),
            'std_ensemble_confidence': results_df['prediction_confidence'].std()
        }
        
        if 'ground_truth' in results_df.columns:
            gt_df = results_df.dropna(subset=['ground_truth'])
            if len(gt_df) > 0:
                ensemble_analysis.update({
                    'accuracy': gt_df['correct'].mean(),
                    'correct_predictions': gt_df['correct'].sum(),
                    'total_with_gt': len(gt_df)
                })
        
        # Save analysis
        output_file = self.analysis_dir / "comparisons" / "ensemble_performance.json"
        with open(output_file, 'w') as f:
            json.dump(ensemble_analysis, f, indent=2, default=str)
        
        return output_file
    
    def _visualize_segmentation_performance(self, perf_df: pd.DataFrame):
        """Create visualization for segmentation type performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Mean confidence comparison
        axes[0,0].bar(perf_df['segmentation_type'], perf_df['mean_confidence'])
        axes[0,0].set_title('Mean Confidence by Segmentation Type')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Confidence variability
        axes[0,1].bar(perf_df['segmentation_type'], perf_df['std_confidence'])
        axes[0,1].set_title('Confidence Variability (Std Dev)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # High confidence predictions
        axes[1,0].bar(perf_df['segmentation_type'], perf_df['high_confidence_count'])
        axes[1,0].set_title('High Confidence Predictions (>0.8)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Above threshold predictions
        axes[1,1].bar(perf_df['segmentation_type'], perf_df['above_threshold_count'])
        axes[1,1].set_title('Above Threshold Predictions (>0.5)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "comparisons" / "segmentation_performance_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_disease_patterns(self, analysis_df: pd.DataFrame):
        """Visualize disease prediction patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Disease prediction frequency
        top_diseases = analysis_df.head(10)
        axes[0,0].barh(top_diseases['disease'], top_diseases['times_predicted'])
        axes[0,0].set_title('Disease Prediction Frequency (Top 10)')
        
        # Mean confidence by disease
        axes[0,1].barh(top_diseases['disease'], top_diseases['mean_confidence'])
        axes[0,1].set_title('Mean Confidence by Disease')
        
        # Prediction rate
        axes[1,0].barh(top_diseases['disease'], top_diseases['prediction_rate'])
        axes[1,0].set_title('Disease Prediction Rate')
        
        # High confidence instances
        axes[1,1].barh(top_diseases['disease'], top_diseases['high_confidence_instances'])
        axes[1,1].set_title('High Confidence Instances')
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "comparisons" / "disease_pattern_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_consistency_accuracy(self, merged_df: pd.DataFrame):
        """Visualize consistency vs accuracy relationship"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot: consistency vs accuracy
        axes[0].scatter(merged_df['overall_consistency'], merged_df['correct'])
        axes[0].set_xlabel('Overall Consistency')
        axes[0].set_ylabel('Correct Prediction (0/1)')
        axes[0].set_title('Consistency vs Accuracy')
        
        # Box plot: accuracy by consistency bins
        consistency_bins = pd.cut(merged_df['overall_consistency'], bins=5)
        merged_df['consistency_bin'] = consistency_bins
        
        box_data = [merged_df[merged_df['consistency_bin'] == cat]['correct'].values 
                   for cat in consistency_bins.categories]
        
        axes[1].boxplot(box_data, labels=[f'Bin {i+1}' for i in range(len(box_data))])
        axes[1].set_xlabel('Consistency Bins (Low to High)')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy Distribution by Consistency')
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "comparisons" / "consistency_accuracy_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_confusion_matrix(self, confusion_pct: pd.DataFrame, labels: List[str]):
        """Visualize confusion matrix"""
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(confusion_pct, 
                   annot=True, 
                   fmt='.1f', 
                   cmap='Blues',
                   xticklabels=labels,
                   yticklabels=labels,
                   cbar_kws={'label': 'Percentage (%)'})
        
        plt.title('ConceptCLIP Confusion Matrix (%)')
        plt.xlabel('Predicted Disease')
        plt.ylabel('Actual Disease')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "matrices" / "confusion_matrix_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        
        # Load all analysis files
        summary = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'total_images_analyzed': 0,
            'available_analyses': []
        }
        
        # Check what analyses are available
        if (self.analysis_dir / "conceptclip_simplified_results.csv").exists():
            df = pd.read_csv(self.analysis_dir / "conceptclip_simplified_results.csv")
            summary['total_images_analyzed'] = len(df)
            summary['available_analyses'].append('simplified_results')
        
        # List all available analysis files
        for analysis_type in ['comparisons', 'matrices', 'individual_outputs']:
            analysis_path = self.analysis_dir / analysis_type
            if analysis_path.exists():
                files = list(analysis_path.glob("*"))
                if files:
                    summary[f'{analysis_type}_files'] = [str(f.name) for f in files]
        
        # Save summary
        with open(self.analysis_dir / "analysis_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary