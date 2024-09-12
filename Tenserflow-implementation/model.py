import tensorflow as tf
from transformers import TFWav2Vec2Model
from typing import Tuple, List, Optional
import numpy as np

class LSLM(tf.keras.Model):
    """
    Listening and Speaking Language Model (LSLM) class.
    This model integrates speaking and listening capabilities using advanced
    neural network architectures.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        num_audio_tokens: int,
        fusion_type: str = 'middle'
    ):
        super(LSLM, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_audio_tokens = num_audio_tokens

        # Model components
        self.speaking_encoder = SpeakingEncoder(vocab_size, d_model, num_audio_tokens)
        self.listening_encoder = ListeningEncoder(d_model)
        self.fusion_module = FusionModule(d_model, fusion_type)
        self.decoder = Decoder(vocab_size, d_model, nhead, num_layers)
        self.turn_taking_detector = TurnTakingDetector(d_model)
        self.vocoder = Vocoder(d_model, num_audio_tokens)

        # Special token for interrupt requests (IRQ)
        self.irq_token = vocab_size  # Assuming IRQ token is at the end of the vocabulary

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training=False):
        """
        Forward pass of the model.

        Args:
            inputs: A tuple containing speaking_input and listening_input.
            training: Boolean indicating whether the call is during training.

        Returns:
            Output logits or generated tokens.
        """
        speaking_input, listening_input = inputs

        # Encode speaking and listening inputs
        speaking_features = self.speaking_encoder(speaking_input, training=training)
        listening_features = self.listening_encoder(listening_input, training=training)

        # Fuse features
        fused_features = self.fusion_module([speaking_features, listening_features], training=training)

        # Decode fused features
        output = self.decoder(fused_features, training=training)

        if not training:
            # Turn-taking logic during inference
            turn_taking_prob = self.turn_taking_detector(listening_features, training=False)
            if tf.reduce_mean(turn_taking_prob) > 0.5:
                # Return IRQ token if turn-taking condition is met
                batch_size = tf.shape(speaking_input)[0]
                return tf.fill([batch_size, 1], self.irq_token)
        return output

    def generate(self, context: tf.Tensor, max_length: int = 1000):
        """
        Generates speech based on the given context.

        Args:
            context: Tensor containing the initial context tokens.
            max_length: Maximum length of the generated sequence.

        Returns:
            Generated audio waveform.
        """
        batch_size = tf.shape(context)[0]
        generated_tokens = []
        for _ in tf.range(max_length):
            # No listening input during generation; use zeros
            listening_input = tf.zeros((batch_size, 1, self.d_model))
            output_logits = self.call((context, listening_input), training=False)
            predicted_token = tf.argmax(output_logits, axis=-1)[:, -1:]  # Get the last token
            generated_tokens.append(predicted_token)

            # Check for IRQ token
            if tf.reduce_all(predicted_token == self.irq_token):
                break

            # Update context
            context = tf.concat([context, predicted_token], axis=1)

        # Concatenate all generated tokens
        generated_sequence = tf.concat(generated_tokens, axis=1)  # [batch_size, seq_len]
        # Generate audio from tokens using vocoder
        generated_audio = self.vocoder(generated_sequence, training=False)
        return generated_audio

class SpeakingEncoder(tf.keras.layers.Layer):
    """
    Encodes speaking inputs (text or audio) into feature representations.
    """
    def __init__(self, vocab_size: int, d_model: int, num_audio_tokens: int):
        super(SpeakingEncoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size + 1, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.audio_quantizer = AudioQuantizer(num_audio_tokens, d_model)

    def call(self, x, training=False):
        if x.dtype.is_integer:
            # Text input (token IDs)
            x = self.embedding(x)
        else:
            # Audio input
            x = self.audio_quantizer(x, training=training)
        x = self.positional_encoding(x)
        return x

class AudioQuantizer(tf.keras.layers.Layer):
    """
    Quantizes audio features into discrete tokens using vector quantization.
    """
    def __init__(self, num_tokens: int, d_model: int):
        super(AudioQuantizer, self).__init__()
        self.num_tokens = num_tokens
        self.d_model = d_model
        # Codebook for vector quantization
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.codebook = self.add_weight(
            shape=(num_tokens, d_model),
            initializer=initializer,
            trainable=True,
            name='codebook'
        )

    def call(self, x, training=False):
        """
        Quantize the input vectors.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model].
            training: Boolean indicating whether the call is during training.

        Returns:
            Quantized embeddings.
        """
        # Flatten batch and sequence dimensions
        flat_x = tf.reshape(x, [-1, self.d_model])  # [batch_size * seq_len, d_model]
        distances = (
            tf.reduce_sum(flat_x**2, axis=1, keepdims=True)
            - 2 * tf.matmul(flat_x, self.codebook, transpose_b=True)
            + tf.reduce_sum(self.codebook**2, axis=1)
        )  # [batch_size * seq_len, num_tokens]

        # Get closest codebook vectors
        encoding_indices = tf.argmin(distances, axis=1)
        quantized = tf.nn.embedding_lookup(self.codebook, encoding_indices)

        # Reshape back to original dimensions
        quantized = tf.reshape(quantized, tf.shape(x))

        if training:
            # Straight-through estimator
            quantized = x + tf.stop_gradient(quantized - x)
        return quantized

class ListeningEncoder(tf.keras.layers.Layer):
    """
    Encodes listening inputs (audio waveforms) into feature representations.
    """
    def __init__(self, d_model: int):
        super(ListeningEncoder, self).__init__()
        # Pretrained Wav2Vec2 model
        self.wav2vec = TFWav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.projection = tf.keras.layers.Dense(d_model)

    def call(self, x, training=False):
        """
        Args:
            x: Input tensor of shape [batch_size, audio_length].
            training: Boolean indicating whether the call is during training.

        Returns:
            Feature representations of shape [batch_size, seq_len, d_model].
        """
        # Extract hidden states from Wav2Vec2
        hidden_states = self.wav2vec(x, training=training).last_hidden_state
        # Project to desired dimension
        x = self.projection(hidden_states)
        return x

class FusionModule(tf.keras.layers.Layer):
    """
    Fuses speaking and listening features using specified fusion strategy.
    """
    def __init__(self, d_model: int, fusion_type: str = 'middle'):
        super(FusionModule, self).__init__()
        self.d_model = d_model
        self.fusion_type = fusion_type.lower()

        if self.fusion_type == 'early':
            self.fusion_layer = tf.keras.layers.Dense(d_model)
        elif self.fusion_type == 'middle':
            self.fusion_layers = [tf.keras.layers.Dense(d_model) for _ in range(6)]
        elif self.fusion_type == 'late':
            self.fusion_layer = tf.keras.layers.Dense(d_model)
        else:
            raise ValueError(f"Invalid fusion_type '{fusion_type}'. Choose from 'early', 'middle', or 'late'.")

    def call(self, inputs: List[tf.Tensor], training=False):
        """
        Args:
            inputs: List containing speaking_features and listening_features.
            training: Boolean indicating whether the call is during training.

        Returns:
            Fused feature representations.
        """
        speaking_features, listening_features = inputs
        if self.fusion_type == 'early':
            fused = self.fusion_layer(tf.concat([speaking_features, listening_features], axis=-1))
        elif self.fusion_type == 'middle':
            fused = speaking_features
            for layer in self.fusion_layers:
                fused = layer(tf.concat([fused, listening_features], axis=-1))
        elif self.fusion_type == 'late':
            fused = self.fusion_layer(tf.concat([speaking_features, listening_features], axis=-1))
        return fused

class Decoder(tf.keras.layers.Layer):
    """
    Decodes fused features into output tokens.
    """
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int):
        super(Decoder, self).__init__()
        self.layers = [DecoderLayer(d_model, nhead) for _ in range(num_layers)]
        self.output_layer = tf.keras.layers.Dense(vocab_size + 1)  # +1 for IRQ token

    def call(self, x, training=False):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model].
            training: Boolean indicating whether the call is during training.

        Returns:
            Logits of shape [batch_size, seq_len, vocab_size + 1].
        """
        for layer in self.layers:
            x = layer(x, training=training)
        logits = self.output_layer(x)
        return logits

class DecoderLayer(tf.keras.layers.Layer):
    """
    A single layer of the Transformer decoder.
    """
    def __init__(self, d_model: int, nhead: int):
        super(DecoderLayer, self).__init__()
        self.self_attention = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model)
        self.feed_forward = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model * 4, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(0.1)

    def call(self, x, training=False):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model].
            training: Boolean indicating whether the call is during training.

        Returns:
            Output tensor of shape [batch_size, seq_len, d_model].
        """
        attn_output = self.self_attention(x, x, training=training)
        x = self.layer_norm1(x + self.dropout(attn_output, training=training))
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output, training=training))
        return x

class TurnTakingDetector(tf.keras.layers.Layer):
    """
    Detects if it's the model's turn to speak.
    """
    def __init__(self, d_model: int):
        super(TurnTakingDetector, self).__init__()
        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(d_model // 2, return_sequences=True)
        )
        self.fc = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x, training=False):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model].
            training: Boolean indicating whether the call is during training.

        Returns:
            Turn-taking probabilities of shape [batch_size, seq_len, 1].
        """
        x = self.lstm(x, training=training)
        x = self.fc(x)
        return x  # [batch_size, seq_len, 1]

class PositionalEncoding(tf.keras.layers.Layer):
    """
    Adds positional encoding to the input embeddings.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        # Create positional encoding matrix
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(-np.arange(0, d_model, 2) * (np.log(10000.0) / d_model))
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.constant(pe, dtype=tf.float32)

    def call(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            Positional encoded tensor of the same shape.
        """
        seq_len = tf.shape(x)[1]
        x = x + self.pe[:seq_len]
        return x

class Vocoder(tf.keras.layers.Layer):
    """
    Converts token sequences into audio waveforms.
    """
    def __init__(self, d_model: int, num_audio_tokens: int):
        super(Vocoder, self).__init__()
        self.prenet = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(d_model // 2, return_sequences=True)
        )
        self.postnet = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=d_model, kernel_size=5, padding='same', activation='tanh'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Conv1D(filters=d_model, kernel_size=5, padding='same', activation='tanh'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Conv1D(filters=num_audio_tokens, kernel_size=5, padding='same')
        ])

    def call(self, x, training=False):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len].
            training: Boolean indicating whether the call is during training.

        Returns:
            Generated audio waveform tensor.
        """
        x = self.prenet(x)
        x = self.lstm(x, training=training)
        x = self.postnet(x, training=training)
        return x  # Output audio waveform or features
