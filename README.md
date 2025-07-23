# Segment-Level Pitch Detection for Singing Voice

A deep learning approach to extract representative pitch values from singing voice audio segments for robust music transcription.

## Overview

This project implements a segment-level pitch detection system that outputs a single representative pitch per audio segment, rather than frame-by-frame estimates. This approach provides more stable pitch contours and is better suited for symbolic music transcription applications.

### Key Features

- **Multiple Model Architectures**: CNN, LSTM, CNN+LSTM, and Transformer-based models
- **Rich Feature Extraction**: Mel-spectrograms, Constant-Q Transform, and combined features
- **Comprehensive Evaluation**: Musical accuracy metrics (cents-based), baseline comparisons
- **Segment-Level Approach**: Reduces pitch jitter compared to frame-wise methods
- **Production Ready**: Complete training, evaluation, and inference pipeline

## Dataset

The project uses the **MIR-1K dataset**, which contains:
- 1000+ singing voice segments from Chinese karaoke recordings
- Frame-by-frame pitch annotations (10ms resolution)
- Voice activity detection labels
- Mixed at different signal-to-accompaniment ratios

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd segment-pitch-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the MIR-1K dataset in the project directory:
```
project/
├── MIR-1K/
│   ├── Wavfile/
│   ├── PitchLabel/
│   ├── vocal-nonvocalLabel/
│   └── readme.txt
```

## Usage

### Quick Start

Run the main training notebook:
```bash
jupyter notebook main_training_notebook.ipynb
```

### Training Custom Models

```python
from src.training import PitchDetectionTrainer

# Initialize trainer
trainer = PitchDetectionTrainer(
    data_root="MIR-1K",
    model_type="cnn_lstm",
    feature_type="mel"
)

# Load and prepare data
trainer.load_and_prepare_data(max_files=100)

# Create and train model
trainer.create_model()
trainer.train(epochs=50, batch_size=32)

# Evaluate
metrics = trainer.evaluate()
print(f"RMSE: {metrics['rmse']:.2f} Hz")
print(f"Cents Accuracy (50): {metrics['cents_accuracy_50']:.1f}%")
```

### Evaluation and Comparison

```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator()
evaluator.add_model_results("Model1", y_true, y_pred)
evaluator.plot_prediction_comparison()

# Generate report
report = create_evaluation_report(evaluator)
print(report)
```

## Model Architectures

### 1. CNN Model
- 2D convolutional layers for spectro-temporal feature extraction
- Global pooling for segment-level aggregation
- Dense layers for pitch regression

### 2. LSTM Model
- Bidirectional LSTM for temporal modeling
- Sequence-to-one architecture for segment prediction
- Dropout for regularization

### 3. CNN+LSTM Hybrid
- 1D CNN for local feature extraction
- LSTM for temporal dependencies
- Combined approach leveraging both architectures

### 4. Transformer Model
- Multi-head self-attention mechanism
- Positional encoding for sequence modeling
- Global average pooling for segment aggregation

## Features

### Audio Features
- **Mel-spectrograms**: Perceptually motivated frequency representation
- **Constant-Q Transform (CQT)**: Logarithmic frequency spacing ideal for music
- **Chroma features**: Pitch class representation
- **Spectral features**: Centroid, rolloff, zero-crossing rate

### Evaluation Metrics
- **RMSE/MAE**: Standard regression metrics
- **Cents Accuracy**: Musical interval-based accuracy (25, 50, 100 cents)
- **R² Score**: Coefficient of determination
- **Octave Error Rate**: Percentage of octave mistakes
- **Baseline Comparison**: Performance vs. simple baselines

## Results

Expected performance on MIR-1K dataset:
- **RMSE**: ~15-25 Hz (depending on model)
- **50-cent Accuracy**: ~75-85%
- **100-cent Accuracy**: ~90-95%

### Model Comparison
| Model | RMSE (Hz) | 50-cent Acc (%) | Parameters |
|-------|-----------|-----------------|------------|
| CNN | ~22 | ~78 | ~500K |
| LSTM | ~20 | ~80 | ~300K |
| CNN+LSTM | ~18 | ~83 | ~400K |
| Transformer | ~19 | ~82 | ~600K |

*Note: Actual results may vary based on training configuration*

## Project Structure

```
project/
├── src/                          # Source code
│   ├── __init__.py
│   ├── data_loader.py           # Dataset loading and preprocessing
│   ├── feature_extraction.py    # Audio feature extraction
│   ├── models.py               # Neural network architectures
│   ├── training.py             # Training pipeline
│   └── evaluation.py           # Evaluation and metrics
├── MIR-1K/                     # Dataset (not included)
├── main_training_notebook.ipynb # Main training notebook
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Technical Details

### Segment Creation
- **Segment Length**: 1.0 second (configurable)
- **Hop Length**: 0.5 seconds (50% overlap)
- **Voiced Threshold**: ≥30% voiced frames per segment
- **Ground Truth**: Median pitch of voiced frames

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: MSE with optional cents-based loss
- **Regularization**: Dropout, early stopping
- **Data Split**: 80% train, 20% validation

### Data Augmentation (Future Work)
- Pitch shifting (±2 semitones)
- Time stretching (0.8-1.2x)
- Noise addition
- Frequency masking

## Applications

1. **Automatic Music Transcription**
   - Convert singing to symbolic notation
   - Stable pitch contours for note segmentation

2. **Music Education**
   - Pitch accuracy assessment for singing
   - Real-time feedback systems

3. **Audio Analysis**
   - Melody extraction from polyphonic music
   - Singer identification and analysis

4. **Music Production**
   - Pitch correction preprocessing
   - Vocal harmony analysis

## Advantages over Frame-wise Methods

1. **Reduced Jitter**: More stable pitch estimates
2. **Better for Transcription**: Segment-level notes align with musical structure
3. **Computational Efficiency**: Fewer predictions needed
4. **Robust to Vibrato**: Averages out micro-variations
5. **Musical Relevance**: Segments correspond to musical events

## Future Improvements

### Model Enhancements
- [ ] Multi-task learning (pitch + voicing)
- [ ] Attention mechanisms for segment weighting
- [ ] Pre-training on larger datasets
- [ ] Ensemble methods

### Data and Features
- [ ] Data augmentation pipeline
- [ ] Multi-modal features (audio + text)
- [ ] Cross-dataset evaluation
- [ ] Real-time processing optimization

### Applications
- [ ] Real-time inference system
- [ ] Web-based demo
- [ ] Mobile app integration
- [ ] Plugin for DAWs

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MIR-1K Dataset creators
- TensorFlow and Librosa communities
- Music Information Retrieval research community

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{segment-pitch-detection,
  title={Segment-Level Pitch Detection for Singing Voice in Music Transcription},
  author={[Your Name]},
  year={2025},
  url={[Repository URL]}
}
```

## Contact

- Author: Nguyen Vu Linh
- Email: linhnvse181687@fpt.edu.vn
- University: FPT University

---

**Note**: This is an academic project for DAT301m course. The implementation focuses on educational value and research reproducibility. 