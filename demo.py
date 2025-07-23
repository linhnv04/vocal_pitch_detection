#!/usr/bin/env python3
"""
Demo script for Segment-Level Pitch Detection
This script demonstrates the basic functionality with a small subset of data.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append('src')

from src.data_loader import MIR1KDataLoader
from src.feature_extraction import AudioFeatureExtractor
from src.training import PitchDetectionTrainer
from src.evaluation import ModelEvaluator, create_evaluation_report


def test_data_loading():
    """Test data loading functionality."""
    print("=" * 50)
    print("Testing Data Loading")
    print("=" * 50)
    
    # Initialize data loader
    data_root = "MIR-1K"
    if not os.path.exists(data_root):
        print(f"Error: Dataset directory '{data_root}' not found!")
        print("Please ensure the MIR-1K dataset is in the project directory.")
        return False
    
    data_loader = MIR1KDataLoader(data_root, sample_rate=16000)
    
    # Get file list
    filenames = data_loader.get_file_list()
    print(f"Found {len(filenames)} audio files")
    
    if len(filenames) == 0:
        print("No audio files found in the dataset!")
        return False
    
    # Test loading a single file
    sample_file = filenames[0]
    print(f"Testing with file: {sample_file}")
    
    try:
        # Load audio and labels
        audio, sr = data_loader.load_audio(f"{sample_file}.wav")
        pitch_labels = data_loader.load_pitch_labels(f"{sample_file}.pv")
        vocal_labels = data_loader.load_vocal_labels(f"{sample_file}.vocal")
        
        print(f"Audio length: {len(audio)/sr:.2f} seconds")
        print(f"Pitch frames: {len(pitch_labels)}")
        print(f"Voiced frames: {np.sum(vocal_labels)}/{len(vocal_labels)} ({np.mean(vocal_labels)*100:.1f}%)")
        
        # Test segment creation
        segments = data_loader.create_segments(audio, pitch_labels, vocal_labels)
        print(f"Created {len(segments)} segments")
        
        if len(segments) > 0:
            print(f"Sample segment pitch: {segments[0]['pitch']:.1f} Hz")
            print("✓ Data loading test passed!")
            return True
        else:
            print("No segments created - check voiced threshold")
            return False
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return False


def test_feature_extraction():
    """Test feature extraction functionality."""
    print("\n" + "=" * 50)
    print("Testing Feature Extraction")
    print("=" * 50)
    
    # Create a synthetic audio signal
    duration = 1.0  # seconds
    sample_rate = 16000
    freq = 220.0  # A3 note
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    
    print(f"Created synthetic audio: {duration}s at {freq} Hz")
    
    try:
        # Test feature extraction
        feature_extractor = AudioFeatureExtractor(sample_rate=sample_rate)
        
        mel_features = feature_extractor.extract_mel_spectrogram(audio)
        cqt_features = feature_extractor.extract_cqt(audio)
        combined_features = feature_extractor.extract_combined_features(audio, "combined")
        
        print(f"Mel spectrogram shape: {mel_features.shape}")
        print(f"CQT features shape: {cqt_features.shape}")
        print(f"Combined features shape: {combined_features.shape}")
        print("✓ Feature extraction test passed!")
        return True
        
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return False


def test_model_creation():
    """Test model creation and compilation."""
    print("\n" + "=" * 50)
    print("Testing Model Creation")
    print("=" * 50)
    
    try:
        from src.models import create_model
        
        # Test different model types
        input_shape = (100, 128)  # (time, features)
        model_types = ["cnn", "lstm", "cnn_lstm"]
        
        for model_type in model_types:
            print(f"Creating {model_type} model...")
            model = create_model(
                model_type=model_type,
                input_shape=input_shape,
                regression=True
            )
            
            # Compile model
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            # Test with dummy data
            dummy_input = np.random.random((1,) + input_shape)
            output = model.predict(dummy_input, verbose=0)
            
            print(f"  Input shape: {dummy_input.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Parameters: {model.count_params():,}")
        
        print("✓ Model creation test passed!")
        return True
        
    except Exception as e:
        print(f"Error in model creation: {e}")
        return False


def run_mini_training():
    """Run a minimal training example."""
    print("\n" + "=" * 50)
    print("Running Mini Training Example")
    print("=" * 50)
    
    try:
        # Initialize trainer with minimal settings
        trainer = PitchDetectionTrainer(
            data_root="MIR-1K",
            model_type="cnn",
            feature_type="mel"
        )
        
        print("Loading data (max 5 files for speed)...")
        trainer.load_and_prepare_data(max_files=5, test_size=0.3)
        
        if trainer.X_train is None:
            print("No training data loaded!")
            return False
        
        print(f"Training samples: {len(trainer.X_train)}")
        print(f"Validation samples: {len(trainer.X_val)}")
        
        print("Creating model...")
        trainer.create_model(dropout_rate=0.2)
        
        print("Training for 3 epochs (demo)...")
        history = trainer.train(epochs=3, batch_size=8, patience=5)
        
        print("Evaluating model...")
        metrics = trainer.evaluate()
        
        print(f"RMSE: {metrics['rmse']:.2f} Hz")
        print(f"Cents Accuracy (50): {metrics['cents_accuracy_50']:.1f}%")
        print("✓ Mini training test passed!")
        return True
        
    except Exception as e:
        print(f"Error in training: {e}")
        return False


def main():
    """Run all demo tests."""
    print("Segment-Level Pitch Detection Demo")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("src"):
        print("Error: Please run this script from the project root directory")
        print("Make sure you see the 'src' folder in the current directory")
        return
    
    # Run tests
    tests = [
        ("Data Loading", test_data_loading),
        ("Feature Extraction", test_feature_extraction),
        ("Model Creation", test_model_creation),
        ("Mini Training", run_mini_training)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Demo Results Summary")
    print("=" * 50)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n🎉 All tests passed! The implementation is working correctly.")
        print("\nNext steps:")
        print("1. Run the full training: jupyter notebook main_training_notebook.ipynb")
        print("2. Experiment with different model architectures and features")
        print("3. Try with more training data for better results")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        print("Make sure you have:")
        print("- MIR-1K dataset in the project directory")
        print("- All required dependencies installed (pip install -r requirements.txt)")
        print("- Sufficient disk space and memory")


if __name__ == "__main__":
    main() 