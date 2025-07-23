#!/usr/bin/env python3
"""
Command-line training script for Segment-Level Pitch Detection
Usage: python run_training.py --model cnn_lstm --features mel --epochs 50
"""

import argparse
import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.training import PitchDetectionTrainer, run_training_experiment


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train segment-level pitch detection models"
    )
    
    # Model configuration
    parser.add_argument(
        '--model', 
        type=str, 
        default='cnn_lstm',
        choices=['cnn', 'lstm', 'cnn_lstm', 'transformer'],
        help='Model architecture to use'
    )
    
    parser.add_argument(
        '--features',
        type=str,
        default='mel',
        choices=['mel', 'cqt', 'mfcc', 'combined'],
        help='Feature type to extract'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate for optimizer'
    )
    
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='Dropout rate for regularization'
    )
    
    # Data configuration
    parser.add_argument(
        '--data-root',
        type=str,
        default='MIR-1K',
        help='Path to MIR-1K dataset'
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Maximum number of files to use (for testing)'
    )
    
    parser.add_argument(
        '--segment-length',
        type=float,
        default=1.0,
        help='Length of audio segments in seconds'
    )
    
    parser.add_argument(
        '--hop-length',
        type=float,
        default=0.5,
        help='Hop length between segments in seconds'
    )
    
    # Output configuration
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Name for the experiment'
    )
    
    # Evaluation
    parser.add_argument(
        '--evaluate-only',
        action='store_true',
        help='Only evaluate existing model'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to saved model for evaluation'
    )
    
    # Multiple experiments
    parser.add_argument(
        '--run-experiments',
        action='store_true',
        help='Run multiple model/feature combinations'
    )
    
    return parser.parse_args()


def single_training(args):
    """Train a single model."""
    print(f"Training {args.model} model with {args.features} features")
    print("=" * 60)
    
    # Create experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.model}_{args.features}_{timestamp}"
    
    # Create output directory
    experiment_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = PitchDetectionTrainer(
        data_root=args.data_root,
        model_type=args.model,
        feature_type=args.features,
        segment_length=args.segment_length,
        hop_length=args.hop_length
    )
    
    print("Loading and preparing data...")
    trainer.load_and_prepare_data(
        max_files=args.max_files,
        test_size=0.2,
        normalize_targets=True
    )
    
    print(f"Training samples: {len(trainer.X_train)}")
    print(f"Validation samples: {len(trainer.X_val)}")
    
    print("Creating model...")
    trainer.create_model(dropout_rate=args.dropout)
    
    # Update learning rate if specified
    if args.learning_rate != 0.001:
        trainer.model.optimizer.learning_rate = args.learning_rate
    
    print("Starting training...")
    history = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=15,
        save_best=True,
        model_save_path=experiment_dir
    )
    
    print("Evaluating model...")
    metrics = trainer.evaluate()
    
    # Save results
    results = {
        'experiment_name': args.experiment_name,
        'model_type': args.model,
        'feature_type': args.features,
        'training_params': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'dropout': args.dropout
        },
        'data_params': {
            'segment_length': args.segment_length,
            'hop_length': args.hop_length,
            'max_files': args.max_files
        },
        'metrics': metrics,
        'history': history
    }
    
    # Save configuration and results
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    with open(os.path.join(experiment_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    trainer.save_config(os.path.join(experiment_dir, 'training_config.json'))
    
    # Plot and save training history
    trainer.plot_training_history(
        save_path=os.path.join(experiment_dir, 'training_history.png')
    )
    
    trainer.plot_predictions(
        num_samples=100,
        save_path=os.path.join(experiment_dir, 'predictions.png')
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Experiment: {args.experiment_name}")
    print(f"RMSE: {metrics['rmse']:.2f} Hz")
    print(f"MAE: {metrics['mae']:.2f} Hz")
    print(f"Cents Accuracy (50): {metrics['cents_accuracy_50']:.1f}%")
    print(f"Cents Accuracy (100): {metrics['cents_accuracy_100']:.1f}%")
    print(f"R²: {metrics['r2']:.3f}")
    print(f"\nResults saved to: {experiment_dir}")
    
    return results


def multiple_experiments(args):
    """Run multiple training experiments."""
    print("Running Multiple Experiments")
    print("=" * 60)
    
    model_types = ['cnn', 'lstm', 'cnn_lstm']
    feature_types = ['mel', 'cqt']
    
    print(f"Will train {len(model_types)} models with {len(feature_types)} feature types")
    print(f"Total experiments: {len(model_types) * len(feature_types)}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.output_dir, f"multi_experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Run experiments
    results = run_training_experiment(
        data_root=args.data_root,
        model_types=model_types,
        feature_types=feature_types,
        max_files=args.max_files,
        epochs=args.epochs
    )
    
    # Save combined results
    with open(os.path.join(experiment_dir, 'all_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary
    summary = []
    for name, result in results.items():
        if 'error' not in result:
            summary.append({
                'experiment': name,
                'rmse': result['metrics']['rmse'],
                'cents_acc_50': result['metrics']['cents_accuracy_50'],
                'cents_acc_100': result['metrics']['cents_accuracy_100'],
                'parameters': result['model_params']
            })
    
    # Sort by RMSE (best first)
    summary.sort(key=lambda x: x['rmse'])
    
    print("\n" + "=" * 60)
    print("Experiment Results Summary")
    print("=" * 60)
    print(f"{'Experiment':<15} {'RMSE (Hz)':<10} {'50c Acc (%)':<12} {'100c Acc (%)':<13} {'Params':<10}")
    print("-" * 65)
    
    for result in summary:
        print(f"{result['experiment']:<15} {result['rmse']:<10.2f} "
              f"{result['cents_acc_50']:<12.1f} {result['cents_acc_100']:<13.1f} "
              f"{result['parameters']:<10,}")
    
    # Find best model
    if summary:
        best = summary[0]
        print(f"\nBest model: {best['experiment']}")
        print(f"RMSE: {best['rmse']:.2f} Hz")
        print(f"50-cent accuracy: {best['cents_acc_50']:.1f}%")
    
    print(f"\nAll results saved to: {experiment_dir}")
    
    return results


def main():
    """Main function."""
    args = parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.data_root):
        print(f"Error: Dataset directory '{args.data_root}' not found!")
        print("Please ensure the MIR-1K dataset is in the project directory.")
        sys.exit(1)
    
    # Check if src directory exists
    if not os.path.exists('src'):
        print("Error: Please run this script from the project root directory")
        sys.exit(1)
    
    print("Segment-Level Pitch Detection Training")
    print("=" * 60)
    print(f"Data root: {args.data_root}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max files: {args.max_files or 'All'}")
    print("")
    
    try:
        if args.run_experiments:
            results = multiple_experiments(args)
        else:
            results = single_training(args)
        
        print("\n🎉 Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 