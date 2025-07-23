import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional


class SegmentPitchDetector(keras.Model):
    """Base class for segment-level pitch detection models."""
    
    def __init__(self, num_classes: Optional[int] = None, 
                 regression: bool = True, **kwargs):
        """
        Initialize the model.
        
        Args:
            num_classes: Number of pitch classes (for classification)
            regression: Whether to use regression (True) or classification (False)
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.regression = regression
        
    def call(self, inputs, training=None):
        raise NotImplementedError
        
    def get_output_layer(self):
        """Get appropriate output layer based on task type."""
        if self.regression:
            return layers.Dense(1, activation='linear', name='pitch_output')
        else:
            return layers.Dense(self.num_classes, activation='softmax', name='pitch_classes')


class CNNPitchDetector(SegmentPitchDetector):
    """CNN-based segment-level pitch detector."""
    
    def __init__(self, input_shape: Tuple[int, int], 
                 num_classes: Optional[int] = None,
                 regression: bool = True,
                 dropout_rate: float = 0.3,
                 **kwargs):
        """
        Initialize CNN model.
        
        Args:
            input_shape: Shape of input features (time, frequency)
            num_classes: Number of pitch classes for classification
            regression: Whether to use regression or classification
            dropout_rate: Dropout rate for regularization
        """
        super().__init__(num_classes=num_classes, regression=regression, **kwargs)
        
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', 
                                  input_shape=input_shape + (1,))
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.conv3 = layers.Conv2D(128, (3, 3), activation='relu')
        
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.pool3 = layers.MaxPooling2D((2, 2))
        
        self.flatten = layers.Flatten()
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dense1 = layers.Dense(256, activation='relu')
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(128, activation='relu')
        
        self.output_layer = self.get_output_layer()
    
    def call(self, inputs, training=None):
        # Add channel dimension if needed
        if len(inputs.shape) == 3:
            x = tf.expand_dims(inputs, -1)
        else:
            x = inputs
            
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.pool3(x)
        
        x = self.flatten(x)
        x = self.dropout1(x, training=training)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        x = self.dense2(x)
        
        return self.output_layer(x)


class LSTMPitchDetector(SegmentPitchDetector):
    """LSTM-based segment-level pitch detector."""
    
    def __init__(self, input_shape: Tuple[int, int],
                 num_classes: Optional[int] = None,
                 regression: bool = True,
                 lstm_units: int = 128,
                 dropout_rate: float = 0.3,
                 **kwargs):
        """
        Initialize LSTM model.
        
        Args:
            input_shape: Shape of input features (time, frequency)
            num_classes: Number of pitch classes for classification
            regression: Whether to use regression or classification
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
        """
        super().__init__(num_classes=num_classes, regression=regression, **kwargs)
        
        self.lstm1 = layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate)
        self.lstm2 = layers.LSTM(lstm_units//2, return_sequences=False, dropout=dropout_rate)
        
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(64, activation='relu')
        
        self.output_layer = self.get_output_layer()
    
    def call(self, inputs, training=None):
        x = self.lstm1(inputs, training=training)
        x = self.lstm2(x, training=training)
        
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        
        return self.output_layer(x)


class CNNLSTMPitchDetector(SegmentPitchDetector):
    """Combined CNN+LSTM model for segment-level pitch detection."""
    
    def __init__(self, input_shape: Tuple[int, int],
                 num_classes: Optional[int] = None,
                 regression: bool = True,
                 lstm_units: int = 64,
                 dropout_rate: float = 0.3,
                 **kwargs):
        """
        Initialize CNN+LSTM model.
        
        Args:
            input_shape: Shape of input features (time, frequency)
            num_classes: Number of pitch classes for classification
            regression: Whether to use regression or classification
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
        """
        super().__init__(num_classes=num_classes, regression=regression, **kwargs)
        
        # CNN feature extraction layers
        self.conv1 = layers.Conv1D(64, 3, activation='relu')
        self.conv2 = layers.Conv1D(128, 3, activation='relu')
        self.pool = layers.MaxPooling1D(2)
        self.dropout1 = layers.Dropout(dropout_rate)
        
        # LSTM temporal modeling layers
        self.lstm1 = layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate)
        self.lstm2 = layers.LSTM(lstm_units//2, return_sequences=False, dropout=dropout_rate)
        
        # Dense layers
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(64, activation='relu')
        
        self.output_layer = self.get_output_layer()
    
    def call(self, inputs, training=None):
        # CNN feature extraction
        x = self.conv1(inputs)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout1(x, training=training)
        
        # LSTM temporal modeling
        x = self.lstm1(x, training=training)
        x = self.lstm2(x, training=training)
        
        # Dense prediction layers
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        x = self.dense2(x)
        
        return self.output_layer(x)


class TransformerPitchDetector(SegmentPitchDetector):
    """Transformer-based segment-level pitch detector."""
    
    def __init__(self, input_shape: Tuple[int, int],
                 num_classes: Optional[int] = None,
                 regression: bool = True,
                 d_model: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout_rate: float = 0.1,
                 **kwargs):
        """
        Initialize Transformer model.
        
        Args:
            input_shape: Shape of input features (time, frequency)
            num_classes: Number of pitch classes for classification
            regression: Whether to use regression or classification
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout_rate: Dropout rate for regularization
        """
        super().__init__(num_classes=num_classes, regression=regression, **kwargs)
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = layers.Dense(d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, num_heads, dropout_rate)
            for _ in range(num_layers)
        ]
        
        # Global average pooling and output
        self.global_pool = layers.GlobalAveragePooling1D()
        self.dropout = layers.Dropout(dropout_rate)
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        
        self.output_layer = self.get_output_layer()
    
    def call(self, inputs, training=None):
        # Input projection
        x = self.input_projection(inputs)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer layers
        for layer in self.encoder_layers:
            x = layer(x, training=training)
        
        # Global pooling and prediction
        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        x = self.dense1(x)
        x = self.dense2(x)
        
        return self.output_layer(x)


class PositionalEncoding(layers.Layer):
    """Positional encoding layer for transformer."""
    
    def __init__(self, d_model, max_length=5000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        position = np.arange(max_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((max_length, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = tf.constant(pe, dtype=tf.float32)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:seq_len, :]


class TransformerEncoderLayer(layers.Layer):
    """Transformer encoder layer."""
    
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super().__init__()
        
        self.multihead_attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model//num_heads
        )
        self.ffn = keras.Sequential([
            layers.Dense(d_model * 4, activation='relu'),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, x, training=None):
        # Multi-head self-attention
        attn_output = self.multihead_attn(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


def create_model(model_type: str, input_shape: Tuple[int, int],
                regression: bool = True, num_classes: Optional[int] = None,
                **kwargs) -> SegmentPitchDetector:
    """
    Create a pitch detection model.
    
    Args:
        model_type: Type of model ("cnn", "lstm", "cnn_lstm", "transformer")
        input_shape: Shape of input features (time, frequency)
        regression: Whether to use regression or classification
        num_classes: Number of classes for classification
        **kwargs: Additional model parameters
        
    Returns:
        Initialized model
    """
    if model_type == "cnn":
        return CNNPitchDetector(
            input_shape=input_shape,
            regression=regression,
            num_classes=num_classes,
            **kwargs
        )
    elif model_type == "lstm":
        return LSTMPitchDetector(
            input_shape=input_shape,
            regression=regression,
            num_classes=num_classes,
            **kwargs
        )
    elif model_type == "cnn_lstm":
        return CNNLSTMPitchDetector(
            input_shape=input_shape,
            regression=regression,
            num_classes=num_classes,
            **kwargs
        )
    elif model_type == "transformer":
        return TransformerPitchDetector(
            input_shape=input_shape,
            regression=regression,
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Custom loss functions
def pitch_mse_loss(y_true, y_pred):
    """Mean squared error loss for pitch regression."""
    return tf.keras.losses.mse(y_true, y_pred)


def pitch_mae_loss(y_true, y_pred):
    """Mean absolute error loss for pitch regression."""
    return tf.keras.losses.mae(y_true, y_pred)


def cents_loss(y_true, y_pred, reference_freq=440.0):
    """Loss function in cents (musical interval)."""
    # Convert to cents
    y_true_cents = 1200 * tf.math.log(y_true / reference_freq) / tf.math.log(2.0)
    y_pred_cents = 1200 * tf.math.log(y_pred / reference_freq) / tf.math.log(2.0)
    
    return tf.keras.losses.mean_squared_error(y_true_cents, y_pred_cents)


# Custom metrics
class CentsAccuracy(keras.metrics.Metric):
    """Accuracy metric in cents."""
    
    def __init__(self, threshold=50.0, reference_freq=440.0, name='cents_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.reference_freq = reference_freq
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert to cents
        y_true_cents = 1200 * tf.math.log(y_true / self.reference_freq) / tf.math.log(2.0)
        y_pred_cents = 1200 * tf.math.log(y_pred / self.reference_freq) / tf.math.log(2.0)
        
        # Calculate absolute difference in cents
        diff = tf.abs(y_true_cents - y_pred_cents)
        
        # Count predictions within threshold
        correct = tf.cast(diff <= self.threshold, tf.float32)
        
        self.total.assign_add(tf.reduce_sum(correct))
        self.count.assign_add(tf.cast(tf.size(correct), tf.float32))
    
    def result(self):
        return self.total / self.count
    
    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0) 