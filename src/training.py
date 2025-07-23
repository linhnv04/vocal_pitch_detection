import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import json
from typing import Tuple, Dict, Any, Optional
from tqdm import tqdm

from .data_loader import MIR1KDataLoader, hz_to_cents, cents_to_hz
from .feature_extraction import AudioFeatureExtractor, prepare_batch_features
from .models import create_model, pitch_mse_loss, pitch_mae_loss, cents_loss, CentsAccuracy


class PitchDetectionTrainer:
    """Trainer class for segment-level pitch detection models."""
    
    def __init__(self, data_root: str, model_type: str = "cnn_lstm",
                 feature_type: str = "mel", sample_rate: int = 16000,
                 segment_length: float = 1.0, hop_length: float = 0.5):
        """
        Initialize trainer.
        
        Args:
            data_root: Path to MIR-1K dataset
            model_type: Type of model to train
            feature_type: Type of features to extract
            sample_rate: Audio sample rate
            segment_length: Length of audio segments in seconds
            hop_length: Hop length between segments in seconds
        """
        self.data_root = data_root
        self.model_type = model_type
        self.feature_type = feature_type
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.hop_length = hop_length
        
        # Initialize components
        self.data_loader = MIR1KDataLoader(data_root, sample_rate)
        self.feature_extractor = AudioFeatureExtractor(sample_rate)
        
        # Training data
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        
        # Model and training history
        self.model = None
        self.history = None
        self.scaler = None
        
    def load_and_prepare_data(self, test_size: float = 0.2, max_files: Optional[int] = None,
                            normalize_targets: bool = True) -> None:
        """Load and prepare training data."""
        print("Loading dataset...")
        
        # Load raw data
        audio_segments, pitch_targets, metadata = self.data_loader.load_dataset(
            segment_length=self.segment_length,
            hop_length=self.hop_length,
            max_files=max_files
        )
        
        print(f"Loaded {len(audio_segments)} segments")
        print(f"Pitch range: {np.min(pitch_targets):.1f} - {np.max(pitch_targets):.1f} Hz")
        
        # Extract features
        print("Extracting features...")
        X = prepare_batch_features(
            audio_segments, self.feature_extractor, 
            feature_type=self.feature_type, normalize=True
        )
        
        y = np.array(pitch_targets)
        
        # Optionally normalize targets
        if normalize_targets:
            self.scaler = StandardScaler()
            y = self.scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Train/validation split
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Validation samples: {len(self.X_val)}")
        print(f"Feature shape: {self.X_train.shape}")
    
    def create_model(self, **model_kwargs) -> None:
        """Create and compile the model."""
        input_shape = self.X_train.shape[1:]  # (time, features)
        
        self.model = create_model(
            model_type=self.model_type,
            input_shape=input_shape,
            regression=True,
            **model_kwargs
        )
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=pitch_mse_loss,
            metrics=[pitch_mae_loss, CentsAccuracy(threshold=50.0)]
        )
        
        # Build model by calling it with sample data
        _ = self.model(self.X_train[:1])
        
        print(f"Created {self.model_type} model")
        print(f"Total parameters: {self.model.count_params():,}")
    
    def train(self, epochs: int = 100, batch_size: int = 32, 
              patience: int = 15, save_best: bool = True,
              model_save_path: str = "models/") -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            patience: Early stopping patience
            save_best: Whether to save the best model
            model_save_path: Path to save models
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        # Create save directory
        os.makedirs(model_save_path, exist_ok=True)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=patience, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=patience//2, min_lr=1e-6
            )
        ]
        
        if save_best:
            model_path = os.path.join(model_save_path, f"best_{self.model_type}_{self.feature_type}.h5")
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    model_path, monitor='val_loss', save_best_only=True
                )
            )
        
        # Train model
        print("Starting training...")
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return self.history.history
    
    def evaluate(self, X_test: Optional[np.ndarray] = None, 
                y_test: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate the model."""
        if X_test is None or y_test is None:
            X_test, y_test = self.X_val, self.y_val
        
        # Get predictions
        y_pred = self.model.predict(X_test, verbose=0)
        
        # Denormalize if necessary
        if self.scaler is not None:
            y_test_orig = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_orig = self.scaler.inverse_transform(y_pred).flatten()
        else:
            y_test_orig = y_test
            y_pred_orig = y_pred.flatten()
        
        # Calculate metrics
        mse = np.mean((y_test_orig - y_pred_orig) ** 2)
        mae = np.mean(np.abs(y_test_orig - y_pred_orig))
        
        # Cents accuracy
        cents_diff = np.abs(hz_to_cents(y_test_orig) - hz_to_cents(y_pred_orig))
        cents_acc_50 = np.mean(cents_diff <= 50) * 100
        cents_acc_100 = np.mean(cents_diff <= 100) * 100
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse),
            'cents_accuracy_50': cents_acc_50,
            'cents_accuracy_100': cents_acc_100,
            'mean_cents_error': np.mean(cents_diff)
        }
        
        return metrics
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """Plot training history."""
        if self.history is None:
            raise ValueError("No training history available.")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE
        axes[0, 1].plot(self.history.history['pitch_mae_loss'], label='Training MAE')
        axes[0, 1].plot(self.history.history['val_pitch_mae_loss'], label='Validation MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Cents accuracy
        if 'cents_accuracy' in self.history.history:
            axes[1, 0].plot(self.history.history['cents_accuracy'], label='Training Accuracy')
            axes[1, 0].plot(self.history.history['val_cents_accuracy'], label='Validation Accuracy')
            axes[1, 0].set_title('Cents Accuracy (50 cents threshold)')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning rate
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_predictions(self, num_samples: int = 50, save_path: Optional[str] = None) -> None:
        """Plot prediction vs ground truth scatter plot."""
        # Get predictions on validation set
        y_pred = self.model.predict(self.X_val[:num_samples], verbose=0)
        
        # Denormalize if necessary
        if self.scaler is not None:
            y_val_orig = self.scaler.inverse_transform(self.y_val[:num_samples].reshape(-1, 1)).flatten()
            y_pred_orig = self.scaler.inverse_transform(y_pred).flatten()
        else:
            y_val_orig = self.y_val[:num_samples]
            y_pred_orig = y_pred.flatten()
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(y_val_orig, y_pred_orig, alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(np.min(y_val_orig), np.min(y_pred_orig))
        max_val = max(np.max(y_val_orig), np.max(y_pred_orig))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        plt.xlabel('Ground Truth Pitch (Hz)')
        plt.ylabel('Predicted Pitch (Hz)')
        plt.title(f'Pitch Predictions vs Ground Truth ({num_samples} samples)')
        plt.grid(True, alpha=0.3)
        
        # Add metrics to plot
        metrics = self.evaluate(self.X_val[:num_samples], self.y_val[:num_samples])
        textstr = f"RMSE: {metrics['rmse']:.2f} Hz\n"
        textstr += f"Cents Acc (50): {metrics['cents_accuracy_50']:.1f}%"
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_config(self, config_path: str) -> None:
        """Save training configuration."""
        config = {
            'model_type': self.model_type,
            'feature_type': self.feature_type,
            'sample_rate': self.sample_rate,
            'segment_length': self.segment_length,
            'hop_length': self.hop_length,
            'input_shape': self.X_train.shape[1:] if self.X_train is not None else None,
            'scaler_mean': self.scaler.mean_.tolist() if self.scaler is not None else None,
            'scaler_scale': self.scaler.scale_.tolist() if self.scaler is not None else None
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)


def run_training_experiment(data_root: str, model_types: list, feature_types: list,
                           max_files: Optional[int] = None, epochs: int = 50) -> Dict[str, Any]:
    """
    Run training experiments with different model and feature combinations.
    
    Args:
        data_root: Path to dataset
        model_types: List of model types to test
        feature_types: List of feature types to test
        max_files: Maximum number of files to use
        epochs: Number of training epochs
        
    Returns:
        Experiment results
    """
    results = {}
    
    for model_type in model_types:
        for feature_type in feature_types:
            experiment_name = f"{model_type}_{feature_type}"
            print(f"\n=== Training {experiment_name} ===")
            
            try:
                # Initialize trainer
                trainer = PitchDetectionTrainer(
                    data_root=data_root,
                    model_type=model_type,
                    feature_type=feature_type
                )
                
                # Load data
                trainer.load_and_prepare_data(max_files=max_files)
                
                # Create and train model
                trainer.create_model()
                history = trainer.train(epochs=epochs)
                
                # Evaluate model
                metrics = trainer.evaluate()
                
                # Save results
                results[experiment_name] = {
                    'history': history,
                    'metrics': metrics,
                    'model_params': trainer.model.count_params()
                }
                
                print(f"Results for {experiment_name}:")
                print(f"  RMSE: {metrics['rmse']:.2f} Hz")
                print(f"  Cents Accuracy (50): {metrics['cents_accuracy_50']:.1f}%")
                
            except Exception as e:
                print(f"Error training {experiment_name}: {e}")
                results[experiment_name] = {'error': str(e)}
    
    return results 