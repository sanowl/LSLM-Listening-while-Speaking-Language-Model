import tensorflow as tf
import math
from transformers import TFWav2Vec2Model, Wav2Vec2FeatureExtractor
import numpy as np

class LSLM(tf.keras.Model):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_audio_tokens):
        super(LSLM, self).__init__()
        self.speaking_encoder = SpeakingEncoder(vocab_size, d_model, num_audio_tokens)
        self.listening_encoder = ListeningEncoder(d_model)
        self.fusion_module = FusionModule(d_model)
        self.decoder = Decoder(vocab_size, d_model, nhead, num_layers)
        self.irq_token = vocab_size  # Assume IRQ token is the last token
        self.turn_taking_detector = TurnTakingDetector(d_model)
        self.vocoder = Vocoder(d_model, num_audio_tokens)

    def call(self, speaking_input, listening_input, is_training=True):
        speaking_features = self.speaking_encoder(speaking_input)
        listening_features = self.listening_encoder(listening_input)
        fused_features = self.fusion_module(speaking_features, listening_features)
        output = self.decoder(fused_features)

        if not is_training:
            turn_taking = self.turn_taking_detector(listening_features)
            if tf.reduce_mean(turn_taking) > 0.5:
                return tf.convert_to_tensor([self.irq_token])

        return output

    def generate(self, context, max_length=1000):
        generated = []
        for _ in range(max_length):
            output = self(context, tf.zeros((1, 1, self.listening_encoder.d_model)), is_training=False)
            if tf.reduce_all(output == self.irq_token):
                break
            generated.append(output)
            context = tf.concat([context, tf.expand_dims(output, axis=1)], axis=1)
        return self.vocoder(tf.concat(generated, axis=0))

class SpeakingEncoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, num_audio_tokens):
        super(SpeakingEncoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size + 1, d_model)  # +1 for IRQ token
        self.positional_encoding = PositionalEncoding(d_model)
        self.audio_quantizer = AudioQuantizer(num_audio_tokens, d_model)

    def call(self, x):
        if x.dtype == tf.int32:  # Text input
            x = self.embedding(x)
        else:  # Audio input
            x = self.audio_quantizer(x)
        x = self.positional_encoding(x)
        return x

class AudioQuantizer(tf.keras.layers.Layer):
    def __init__(self, num_tokens, d_model):
        super(AudioQuantizer, self).__init__()
        self.embedding = tf.keras.layers.Embedding(num_tokens, d_model)
        self.codebook = self.add_weight(shape=(num_tokens, d_model),
                                        initializer=tf.keras.initializers.RandomNormal(),
                                        trainable=True, name="codebook")

    def call(self, x):
        distances = tf.norm(tf.expand_dims(x, 1) - tf.expand_dims(self.codebook, 0), axis=-1)
        indices = tf.argmin(distances, axis=-1)
        return self.embedding(indices)

    def quantize(self, x):
        distances = tf.norm(tf.expand_dims(x, 1) - tf.expand_dims(self.codebook, 0), axis=-1)
        indices = tf.argmin(distances, axis=-1)
        return indices

class ListeningEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super(ListeningEncoder, self).__init__()
        self.wav2vec = TFWav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        self.proj = tf.keras.layers.Dense(d_model)

    def call(self, x):
        inputs = self.feature_extractor(x, return_tensors="tf", padding=True)
        hidden_states = self.wav2vec(inputs.input_values).last_hidden_state
        return self.proj(hidden_states)

class FusionModule(tf.keras.layers.Layer):
    def __init__(self, d_model, fusion_type='middle'):
        super(FusionModule, self).__init__()
        self.d_model = d_model
        self.fusion_type = fusion_type
        self.early_fusion = tf.keras.layers.Dense(d_model)
        self.middle_fusion = [tf.keras.layers.Dense(d_model) for _ in range(6)]
        self.late_fusion = tf.keras.layers.Dense(d_model)

    def call(self, speaking_features, listening_features):
        if self.fusion_type == 'early':
            return self.early_fusion(tf.concat([speaking_features, listening_features], axis=-1))
        elif self.fusion_type == 'middle':
            fused = speaking_features
            for layer in self.middle_fusion:
                fused = layer(tf.concat([fused, listening_features], axis=-1))
            return fused
        elif self.fusion_type == 'late':
            return self.late_fusion(tf.concat([speaking_features, listening_features], axis=-1))

class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(Decoder, self).__init__()
        self.transformer_decoder = tf.keras.layers.Transformer(
            num_layers=num_layers,
            num_heads=nhead,
            d_model=d_model,
            dff=d_model * 4,
        )
        self.fc_out = tf.keras.layers.Dense(vocab_size + 1)  # +1 for IRQ token

    def call(self, x):
        x = self.transformer_decoder(x, x)
        return self.fc_out(x)

class TurnTakingDetector(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super(TurnTakingDetector, self).__init__()
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(d_model // 2, return_sequences=True))
        self.fc = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.lstm(x)
        x = self.fc(x)
        return tf.keras.activations.sigmoid(x)

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.convert_to_tensor(pe, dtype=tf.float32)

    def call(self, x):
        return x + self.pe[:tf.shape(x)[1], :]

class Vocoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_audio_tokens):
        super(Vocoder, self).__init__()
        self.prenet = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(d_model)
        ])
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(d_model // 2, return_sequences=True))
        self.postnet = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=d_model, kernel_size=5, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('tanh'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Conv1D(filters=d_model, kernel_size=5, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('tanh'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Conv1D(filters=num_audio_tokens, kernel_size=5, padding='same')
        ])

    def call(self, x):
        x = self.prenet(x)
        x = self.lstm(x)
        x = tf.transpose(x, perm=[0, 2, 1])  # Transpose to match Conv1D input requirements
        x = self.postnet(x)
        return tf.transpose(x, perm=[0, 2, 1])

# Instantiate and test the model
model = LSLM(VOCAB_SIZE=10000, d_model=512, nhead=8, num_layers=6, num_audio_tokens=1024)
