import numpy as np
import librosa
import tensorflow as tf
from typing import Tuple, Optional


class AudioFeatureExtractor:
    """Extract features from audio segments for pitch detection."""
    
    def __init__(self, sample_rate: int = 16000, n_fft: int = 2048, 
                 hop_length: int = 512, n_mels: int = 128, fmin: float = 50.0, 
                 fmax: float = 2000.0):
        """
        Initialize feature extractor.
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel bins
            fmin: Minimum frequency for mel scale
            fmax: Maximum frequency for mel scale
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        
        # Precompute mel filter bank
        self.mel_fb = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, 
            fmin=fmin, fmax=fmax
        )
        
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel-spectrogram features."""
        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Apply mel filter bank
        mel_spec = np.dot(self.mel_fb, magnitude)
        
        # Convert to log scale
        log_mel_spec = librosa.amplitude_to_db(mel_spec + 1e-10)
        
        return log_mel_spec.T  # Shape: (time, freq)
    
    def extract_cqt(self, audio: np.ndarray, n_bins: int = 84) -> np.ndarray:
        """Extract Constant-Q Transform features."""
        # Use a smaller number of bins to avoid frequency issues
        n_bins = min(n_bins, 72)  # Limit to 6 octaves
        
        cqt = librosa.cqt(
            audio, sr=self.sample_rate, hop_length=self.hop_length,
            n_bins=n_bins, fmin=librosa.note_to_hz('C2')
        )
        
        # Convert to magnitude and log scale
        log_cqt = librosa.amplitude_to_db(np.abs(cqt) + 1e-10)
        
        return log_cqt.T  # Shape: (time, freq)
    
    def extract_chroma(self, audio: np.ndarray) -> np.ndarray:
        """Extract chroma features."""
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        return chroma.T  # Shape: (time, 12)
    
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """Extract MFCC features."""
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate, n_mfcc=n_mfcc,
            hop_length=self.hop_length
        )
        return mfcc.T  # Shape: (time, n_mfcc)
    
    def extract_spectral_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract additional spectral features."""
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            y=audio, hop_length=self.hop_length
        )[0]
        
        # Stack features
        features = np.stack([spectral_centroids, spectral_rolloff, zcr], axis=0)
        return features.T  # Shape: (time, 3)
    
    def extract_combined_features(self, audio: np.ndarray, 
                                feature_type: str = "mel") -> np.ndarray:
        """
        Extract combined features for model input.
        
        Args:
            audio: Audio segment
            feature_type: Type of features to extract ("mel", "cqt", "combined")
            
        Returns:
            Feature matrix of shape (time, features)
        """
        if feature_type == "mel":
            return self.extract_mel_spectrogram(audio)
        elif feature_type == "cqt":
            return self.extract_cqt(audio)
        elif feature_type == "mfcc":
            return self.extract_mfcc(audio)
        elif feature_type == "combined":
            # Combine multiple feature types
            mel_spec = self.extract_mel_spectrogram(audio)
            chroma = self.extract_chroma(audio)
            spectral = self.extract_spectral_features(audio)
            
            # Ensure all features have the same time dimension
            min_time = min(mel_spec.shape[0], chroma.shape[0], spectral.shape[0])
            mel_spec = mel_spec[:min_time]
            chroma = chroma[:min_time]
            spectral = spectral[:min_time]
            
            # Concatenate features
            combined = np.concatenate([mel_spec, chroma, spectral], axis=1)
            return combined
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    def normalize_features(self, features: np.ndarray, 
                          method: str = "standard") -> np.ndarray:
        """
        Normalize features.
        
        Args:
            features: Feature matrix
            method: Normalization method ("standard", "minmax", "robust")
        """
        if method == "standard":
            mean = np.mean(features, axis=0, keepdims=True)
            std = np.std(features, axis=0, keepdims=True) + 1e-8
            return (features - mean) / std
        elif method == "minmax":
            min_val = np.min(features, axis=0, keepdims=True)
            max_val = np.max(features, axis=0, keepdims=True)
            return (features - min_val) / (max_val - min_val + 1e-8)
        elif method == "robust":
            median = np.median(features, axis=0, keepdims=True)
            mad = np.median(np.abs(features - median), axis=0, keepdims=True)
            return (features - median) / (mad + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")


def pad_or_truncate(features: np.ndarray, target_length: int) -> np.ndarray:
    """Pad or truncate feature sequence to target length."""
    current_length = features.shape[0]
    
    if current_length == target_length:
        return features
    elif current_length < target_length:
        # Pad with zeros
        pad_length = target_length - current_length
        padding = np.zeros((pad_length, features.shape[1]))
        return np.concatenate([features, padding], axis=0)
    else:
        # Truncate
        return features[:target_length]


def prepare_batch_features(audio_segments: list, feature_extractor: AudioFeatureExtractor,
                          feature_type: str = "mel", target_length: Optional[int] = None,
                          normalize: bool = True) -> np.ndarray:
    """
    Prepare a batch of features from audio segments.
    
    Args:
        audio_segments: List of audio segments
        feature_extractor: Feature extractor instance
        feature_type: Type of features to extract
        target_length: Target sequence length (auto-computed if None)
        normalize: Whether to normalize features
        
    Returns:
        Batch of features with shape (batch_size, time, features)
    """
    # Extract features for all segments
    feature_list = []
    for audio in audio_segments:
        features = feature_extractor.extract_combined_features(audio, feature_type)
        
        if normalize:
            features = feature_extractor.normalize_features(features)
            
        feature_list.append(features)
    
    # Determine target length if not provided
    if target_length is None:
        target_length = max(f.shape[0] for f in feature_list)
    
    # Pad/truncate to same length
    padded_features = []
    for features in feature_list:
        padded = pad_or_truncate(features, target_length)
        padded_features.append(padded)
    
    return np.array(padded_features) 