import os
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from typing import List, Tuple, Optional
import glob
from tqdm import tqdm


class MIR1KDataLoader:
    """Data loader for MIR-1K dataset for segment-level pitch detection."""
    
    def __init__(self, data_root: str, sample_rate: int = 16000):
        """
        Initialize the data loader.
        
        Args:
            data_root: Path to MIR-1K dataset root directory
            sample_rate: Target sample rate for audio processing
        """
        self.data_root = data_root
        self.sample_rate = sample_rate
        self.frame_duration = 0.01  # 10ms frames as per dataset specification
        
        # Dataset subdirectories
        self.wavfile_dir = os.path.join(data_root, "Wavfile")
        self.pitch_dir = os.path.join(data_root, "PitchLabel")
        self.vocal_dir = os.path.join(data_root, "vocal-nonvocalLabel")
        
    def load_pitch_labels(self, filename: str) -> np.ndarray:
        """Load pitch labels from .pv file."""
        filepath = os.path.join(self.pitch_dir, filename)
        with open(filepath, 'r') as f:
            pitches = [float(line.strip()) for line in f.readlines()]
        return np.array(pitches)
    
    def load_vocal_labels(self, filename: str) -> np.ndarray:
        """Load vocal/non-vocal labels from .vocal file."""
        filepath = os.path.join(self.vocal_dir, filename)
        with open(filepath, 'r') as f:
            labels = [int(float(line.strip())) for line in f.readlines()]
        return np.array(labels)
    
    def load_audio(self, filename: str) -> Tuple[np.ndarray, int]:
        """Load audio file and resample to target sample rate."""
        filepath = os.path.join(self.wavfile_dir, filename)
        audio, sr = librosa.load(filepath, sr=self.sample_rate)
        return audio, sr
    
    def get_file_list(self) -> List[str]:
        """Get list of all audio files in the dataset."""
        wav_files = glob.glob(os.path.join(self.wavfile_dir, "*.wav"))
        # Extract just the filename without path and extension
        filenames = [os.path.splitext(os.path.basename(f))[0] for f in wav_files]
        return sorted(filenames)
    
    def create_segments(self, audio: np.ndarray, pitch_labels: np.ndarray, 
                       vocal_labels: np.ndarray, segment_length: float = 1.0,
                       hop_length: float = 0.5) -> List[dict]:
        """
        Create segments from audio and labels.
        
        Args:
            audio: Audio signal
            pitch_labels: Frame-level pitch labels (10ms frames)
            vocal_labels: Frame-level vocal/non-vocal labels
            segment_length: Length of each segment in seconds
            hop_length: Hop length between segments in seconds
            
        Returns:
            List of segment dictionaries with audio, pitch, and metadata
        """
        segments = []
        
        # Convert segment parameters to samples/frames
        segment_samples = int(segment_length * self.sample_rate)
        hop_samples = int(hop_length * self.sample_rate)
        
        # Convert to frame indices (10ms frames)
        frames_per_segment = int(segment_length / self.frame_duration)
        frames_per_hop = int(hop_length / self.frame_duration)
        
        num_segments = (len(audio) - segment_samples) // hop_samples + 1
        
        for i in range(num_segments):
            # Audio segment
            start_sample = i * hop_samples
            end_sample = start_sample + segment_samples
            
            if end_sample > len(audio):
                break
                
            audio_segment = audio[start_sample:end_sample]
            
            # Corresponding frame indices
            start_frame = i * frames_per_hop
            end_frame = start_frame + frames_per_segment
            
            if end_frame > len(pitch_labels):
                break
                
            pitch_segment = pitch_labels[start_frame:end_frame]
            vocal_segment = vocal_labels[start_frame:end_frame]
            
            # Calculate segment-level pitch (only from voiced frames)
            voiced_frames = (vocal_segment == 1) & (pitch_segment > 0)
            
            if np.sum(voiced_frames) > frames_per_segment * 0.3:  # At least 30% voiced
                # Use median pitch as segment-level ground truth
                voiced_pitches = pitch_segment[voiced_frames]
                segment_pitch = np.median(voiced_pitches)
                
                segments.append({
                    'audio': audio_segment,
                    'pitch': segment_pitch,
                    'voiced_ratio': np.sum(voiced_frames) / len(voiced_frames),
                    'pitch_std': np.std(voiced_pitches),  # Measure of vibrato/stability
                    'start_time': start_sample / self.sample_rate,
                    'duration': segment_length
                })
        
        return segments
    
    def load_dataset(self, segment_length: float = 1.0, hop_length: float = 0.5,
                    max_files: Optional[int] = None) -> Tuple[List[np.ndarray], List[float], List[dict]]:
        """
        Load the entire dataset and create segments.
        
        Returns:
            audio_segments: List of audio segments
            pitch_targets: List of segment-level pitch targets
            metadata: List of metadata dictionaries
        """
        filenames = self.get_file_list()
        
        if max_files:
            filenames = filenames[:max_files]
        
        audio_segments = []
        pitch_targets = []
        metadata = []
        
        print(f"Loading {len(filenames)} files...")
        
        for filename in tqdm(filenames):
            try:
                # Load data
                audio, _ = self.load_audio(f"{filename}.wav")
                pitch_labels = self.load_pitch_labels(f"{filename}.pv")
                vocal_labels = self.load_vocal_labels(f"{filename}.vocal")
                
                # Create segments
                segments = self.create_segments(audio, pitch_labels, vocal_labels,
                                              segment_length, hop_length)
                
                # Add to dataset
                for segment in segments:
                    audio_segments.append(segment['audio'])
                    pitch_targets.append(segment['pitch'])
                    
                    # Add file info to metadata
                    meta = segment.copy()
                    meta['filename'] = filename
                    del meta['audio']  # Don't duplicate audio data
                    metadata.append(meta)
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        print(f"Created {len(audio_segments)} segments from {len(filenames)} files")
        
        return audio_segments, pitch_targets, metadata


def hz_to_cents(freq, reference: float = 440.0):
    """Convert frequency in Hz to cents relative to reference frequency."""
    freq = np.asarray(freq)
    result = np.zeros_like(freq)
    valid_mask = freq > 0
    result[valid_mask] = 1200 * np.log2(freq[valid_mask] / reference)
    return result


def cents_to_hz(cents: float, reference: float = 440.0) -> float:
    """Convert cents to frequency in Hz relative to reference frequency."""
    return reference * (2 ** (cents / 1200)) 