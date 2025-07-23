import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import r2_score
import librosa

from .data_loader import hz_to_cents, cents_to_hz


class ModelEvaluator:
    """Comprehensive evaluation tools for pitch detection models."""
    
    def __init__(self):
        self.results = {}
    
    def add_model_results(self, model_name: str, y_true: np.ndarray, 
                         y_pred: np.ndarray, metadata: Optional[List[dict]] = None):
        """Add evaluation results for a model."""
        self.results[model_name] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'metadata': metadata or []
        }
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        # Basic regression metrics
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Relative error metrics
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Cents-based metrics
        cents_true = hz_to_cents(y_true)
        cents_pred = hz_to_cents(y_pred)
        cents_diff = np.abs(cents_true - cents_pred)
        
        cents_acc_25 = np.mean(cents_diff <= 25) * 100
        cents_acc_50 = np.mean(cents_diff <= 50) * 100
        cents_acc_100 = np.mean(cents_diff <= 100) * 100
        
        # Octave errors
        octave_errors = np.mean(cents_diff >= 1200) * 100
        
        # Semitone accuracy
        semitone_errors = np.mean(cents_diff >= 100) * 100
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'cents_acc_25': cents_acc_25,
            'cents_acc_50': cents_acc_50,
            'cents_acc_100': cents_acc_100,
            'mean_cents_error': np.mean(cents_diff),
            'median_cents_error': np.median(cents_diff),
            'std_cents_error': np.std(cents_diff),
            'octave_error_rate': octave_errors,
            'semitone_error_rate': semitone_errors
        }
    
    def evaluate_all_models(self) -> pd.DataFrame:
        """Evaluate all models and return metrics DataFrame."""
        metrics_list = []
        
        for model_name, data in self.results.items():
            metrics = self.calculate_metrics(data['y_true'], data['y_pred'])
            metrics['model'] = model_name
            metrics_list.append(metrics)
        
        return pd.DataFrame(metrics_list)
    
    def plot_prediction_comparison(self, figsize: Tuple[int, int] = (15, 10),
                                 save_path: Optional[str] = None):
        """Plot prediction scatter plots for all models."""
        n_models = len(self.results)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (model_name, data) in enumerate(self.results.items()):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            y_true = data['y_true']
            y_pred = data['y_pred']
            
            # Scatter plot
            ax.scatter(y_true, y_pred, alpha=0.6, s=20)
            
            # Perfect prediction line
            min_val = min(np.min(y_true), np.min(y_pred))
            max_val = max(np.max(y_true), np.max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
            
            # Calculate metrics for display
            metrics = self.calculate_metrics(y_true, y_pred)
            
            ax.set_xlabel('Ground Truth Pitch (Hz)')
            ax.set_ylabel('Predicted Pitch (Hz)')
            ax.set_title(f'{model_name}\nRMSE: {metrics["rmse"]:.2f} Hz, '
                        f'Cents Acc (50): {metrics["cents_acc_50"]:.1f}%')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_models, rows * cols):
            row = idx // cols
            col = idx % cols
            if rows > 1:
                axes[row, col].set_visible(False)
            elif cols > 1:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_error_distribution(self, figsize: Tuple[int, int] = (12, 8),
                               save_path: Optional[str] = None):
        """Plot error distribution in cents for all models."""
        plt.figure(figsize=figsize)
        
        for model_name, data in self.results.items():
            y_true = data['y_true']
            y_pred = data['y_pred']
            
            cents_true = hz_to_cents(y_true)
            cents_pred = hz_to_cents(y_pred)
            cents_error = cents_pred - cents_true
            
            plt.hist(cents_error, bins=50, alpha=0.7, label=model_name, density=True)
        
        plt.xlabel('Pitch Error (cents)')
        plt.ylabel('Density')
        plt.title('Distribution of Pitch Errors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.8)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_metrics_comparison(self, figsize: Tuple[int, int] = (15, 10),
                               save_path: Optional[str] = None):
        """Plot comparison of key metrics across models."""
        metrics_df = self.evaluate_all_models()
        
        # Select key metrics for visualization
        key_metrics = ['rmse', 'cents_acc_50', 'cents_acc_100', 'mean_cents_error', 'r2']
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, metric in enumerate(key_metrics):
            ax = axes[idx]
            
            bars = ax.bar(metrics_df['model'], metrics_df[metric])
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom')
        
        # Remove empty subplot
        axes[-1].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_by_pitch_range(self, num_bins: int = 5) -> pd.DataFrame:
        """Analyze model performance by pitch range."""
        results_list = []
        
        for model_name, data in self.results.items():
            y_true = data['y_true']
            y_pred = data['y_pred']
            
            # Create pitch range bins
            pitch_bins = pd.qcut(y_true, q=num_bins, labels=False)
            bin_edges = pd.qcut(y_true, q=num_bins, retbins=True)[1]
            
            for bin_idx in range(num_bins):
                mask = pitch_bins == bin_idx
                if np.sum(mask) > 0:
                    bin_true = y_true[mask]
                    bin_pred = y_pred[mask]
                    
                    metrics = self.calculate_metrics(bin_true, bin_pred)
                    metrics['model'] = model_name
                    metrics['pitch_range'] = f'{bin_edges[bin_idx]:.1f}-{bin_edges[bin_idx+1]:.1f} Hz'
                    metrics['bin_idx'] = bin_idx
                    metrics['num_samples'] = np.sum(mask)
                    
                    results_list.append(metrics)
        
        return pd.DataFrame(results_list)
    
    def analyze_by_vibrato(self, vibrato_threshold: float = 2.0) -> pd.DataFrame:
        """Analyze model performance based on vibrato intensity."""
        if not all('metadata' in data and data['metadata'] for data in self.results.values()):
            print("Warning: Metadata not available for vibrato analysis")
            return pd.DataFrame()
        
        results_list = []
        
        for model_name, data in self.results.items():
            y_true = data['y_true']
            y_pred = data['y_pred']
            metadata = data['metadata']
            
            # Extract pitch standard deviation as vibrato measure
            pitch_stds = np.array([meta.get('pitch_std', 0) for meta in metadata])
            
            # Classify segments as high/low vibrato
            high_vibrato = pitch_stds > vibrato_threshold
            
            for vibrato_level, mask in [('Low Vibrato', ~high_vibrato), 
                                      ('High Vibrato', high_vibrato)]:
                if np.sum(mask) > 0:
                    subset_true = y_true[mask]
                    subset_pred = y_pred[mask]
                    
                    metrics = self.calculate_metrics(subset_true, subset_pred)
                    metrics['model'] = model_name
                    metrics['vibrato_level'] = vibrato_level
                    metrics['num_samples'] = np.sum(mask)
                    
                    results_list.append(metrics)
        
        return pd.DataFrame(results_list)


def compare_with_baseline(y_true: np.ndarray, y_pred: np.ndarray, 
                         baseline_type: str = "median") -> Dict[str, float]:
    """Compare model performance with simple baselines."""
    if baseline_type == "median":
        baseline_pred = np.full_like(y_true, np.median(y_true))
    elif baseline_type == "mean":
        baseline_pred = np.full_like(y_true, np.mean(y_true))
    elif baseline_type == "mode":
        # Use most common pitch (discretized)
        pitch_bins = np.round(y_true).astype(int)
        mode_pitch = np.bincount(pitch_bins).argmax()
        baseline_pred = np.full_like(y_true, mode_pitch)
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")
    
    # Calculate improvement over baseline
    model_mse = np.mean((y_true - y_pred) ** 2)
    baseline_mse = np.mean((y_true - baseline_pred) ** 2)
    
    model_mae = np.mean(np.abs(y_true - y_pred))
    baseline_mae = np.mean(np.abs(y_true - baseline_pred))
    
    return {
        'baseline_type': baseline_type,
        'model_mse': model_mse,
        'baseline_mse': baseline_mse,
        'mse_improvement': (baseline_mse - model_mse) / baseline_mse * 100,
        'model_mae': model_mae,
        'baseline_mae': baseline_mae,
        'mae_improvement': (baseline_mae - model_mae) / baseline_mae * 100
    }


def plot_pitch_contour_comparison(y_true: np.ndarray, y_pred: np.ndarray,
                                segment_indices: List[int], 
                                time_per_segment: float = 1.0,
                                figsize: Tuple[int, int] = (15, 8),
                                save_path: Optional[str] = None):
    """Plot pitch contours for selected segments."""
    n_segments = len(segment_indices)
    
    plt.figure(figsize=figsize)
    
    for i, seg_idx in enumerate(segment_indices):
        time_offset = i * (time_per_segment + 0.5)  # Add gap between segments
        time_points = np.linspace(time_offset, time_offset + time_per_segment, 1)
        
        plt.plot(time_points, [y_true[seg_idx]], 'o-', 
                label='Ground Truth' if i == 0 else '', 
                color='blue', markersize=8)
        plt.plot(time_points, [y_pred[seg_idx]], 's-', 
                label='Prediction' if i == 0 else '', 
                color='red', markersize=8)
        
        # Add segment separator
        if i < n_segments - 1:
            plt.axvline(x=time_offset + time_per_segment + 0.25, 
                       color='gray', linestyle='--', alpha=0.5)
        
        # Add segment index annotation
        plt.text(time_offset + time_per_segment/2, 
                max(y_true[seg_idx], y_pred[seg_idx]) + 20,
                f'Seg {seg_idx}', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Time (segments)')
    plt.ylabel('Pitch (Hz)')
    plt.title('Pitch Contour Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_evaluation_report(evaluator: ModelEvaluator, 
                           save_path: Optional[str] = None) -> str:
    """Create a comprehensive evaluation report."""
    metrics_df = evaluator.evaluate_all_models()
    
    report = []
    report.append("=" * 80)
    report.append("SEGMENT-LEVEL PITCH DETECTION EVALUATION REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Overall metrics
    report.append("OVERALL METRICS")
    report.append("-" * 40)
    for _, row in metrics_df.iterrows():
        report.append(f"\nModel: {row['model']}")
        report.append(f"  RMSE: {row['rmse']:.2f} Hz")
        report.append(f"  MAE: {row['mae']:.2f} Hz")
        report.append(f"  R²: {row['r2']:.3f}")
        report.append(f"  Cents Accuracy (50): {row['cents_acc_50']:.1f}%")
        report.append(f"  Cents Accuracy (100): {row['cents_acc_100']:.1f}%")
        report.append(f"  Mean Cents Error: {row['mean_cents_error']:.1f}")
        report.append(f"  Octave Error Rate: {row['octave_error_rate']:.1f}%")
    
    report.append("\n")
    
    # Best model identification
    best_model_rmse = metrics_df.loc[metrics_df['rmse'].idxmin(), 'model']
    best_model_cents = metrics_df.loc[metrics_df['cents_acc_50'].idxmax(), 'model']
    
    report.append("BEST MODELS")
    report.append("-" * 40)
    report.append(f"Best RMSE: {best_model_rmse}")
    report.append(f"Best Cents Accuracy: {best_model_cents}")
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
    
    return report_text 