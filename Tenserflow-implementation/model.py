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
        self.irq_token = tf.constant(vocab_size, dtype=tf.int32)
        self.turn_taking_detector = TurnTakingDetector(d_model)
        self.vocoder = Vocoder(d_model, num_audio_tokens)

    @tf.function
    def call(self, inputs, training=True):
        speaking_input, listening_input = inputs
        speaking_features = self.speaking_encoder(speaking_input, training=training)
        listening_features = self.listening_encoder(listening_input, training=training)
        fused_features = self.fusion_module([speaking_features, listening_features], training=training)
        output = self.decoder(fused_features, training=training)

        if not training:
            turn_taking = self.turn_taking_detector(listening_features, training=False)
            return tf.where(tf.reduce_mean(turn_taking) > 0.5,
                            tf.expand_dims(self.irq_token, 0),
                            output)
        return output

    @tf.function
    def generate(self, context, max_length=1000):
        batch_size = tf.shape(context)[0]
        generated = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        for i in tf.range(max_length):
            output = self([context, tf.zeros((batch_size, 1, self.listening_encoder.d_model))], training=False)
            generated = generated.write(i, output)
            if tf.reduce_all(output == self.irq_token):
                break
            context = tf.concat([context, tf.expand_dims(output, axis=1)], axis=1)
        return self.vocoder(generated.stack(), training=False)

class SpeakingEncoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, num_audio_tokens):
        super(SpeakingEncoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size + 1, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.audio_quantizer = AudioQuantizer(num_audio_tokens, d_model)

    @tf.function
    def call(self, x, training=True):
        x = tf.cond(tf.equal(tf.rank(x), 2),
                    lambda: self.embedding(x),
                    lambda: self.audio_quantizer(x, training=training))
        return self.positional_encoding(x)

class AudioQuantizer(tf.keras.layers.Layer):
    def __init__(self, num_tokens, d_model):
        super(AudioQuantizer, self).__init__()
        self.embedding = tf.keras.layers.Embedding(num_tokens, d_model)
        self.codebook = self.add_weight(shape=(num_tokens, d_model),
                                        initializer=tf.keras.initializers.GlorotUniform(),
                                        trainable=True, name="codebook")

    @tf.function
    def call(self, x, training=True):
        distances = tf.norm(tf.expand_dims(x, 1) - self.codebook, axis=-1)
        indices = tf.argmin(distances, axis=-1)
        return self.embedding(indices)

    @tf.function
    def quantize(self, x):
        distances = tf.norm(tf.expand_dims(x, 1) - self.codebook, axis=-1)
        return tf.argmin(distances, axis=-1)

class ListeningEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super(ListeningEncoder, self).__init__()
        self.wav2vec = TFWav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        self.proj = tf.keras.layers.Dense(d_model)
        self.d_model = d_model

    @tf.function
    def call(self, x, training=True):
        inputs = self.feature_extractor(x, return_tensors="tf", padding=True)
        hidden_states = self.wav2vec(inputs.input_values, training=training).last_hidden_state
        return self.proj(hidden_states)

class FusionModule(tf.keras.layers.Layer):
    def __init__(self, d_model, fusion_type='middle'):
        super(FusionModule, self).__init__()
        self.d_model = d_model
        self.fusion_type = fusion_type
        self.early_fusion = tf.keras.layers.Dense(d_model)
        self.middle_fusion = tf.keras.Sequential([tf.keras.layers.Dense(d_model) for _ in range(6)])
        self.late_fusion = tf.keras.layers.Dense(d_model)

    @tf.function
    def call(self, inputs, training=True):
        speaking_features, listening_features = inputs
        if self.fusion_type == 'early':
            return self.early_fusion(tf.concat([speaking_features, listening_features], axis=-1))
        elif self.fusion_type == 'middle':
            fused = speaking_features
            for layer in self.middle_fusion.layers:
                fused = layer(tf.concat([fused, listening_features], axis=-1))
            return fused
        elif self.fusion_type == 'late':
            return self.late_fusion(tf.concat([speaking_features, listening_features], axis=-1))

class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(Decoder, self).__init__()
        self.decoder_layers = [DecoderLayer(d_model, nhead) for _ in range(num_layers)]
        self.fc_out = tf.keras.layers.Dense(vocab_size + 1)

    @tf.function
    def call(self, x, training=True):
        for layer in self.decoder_layers:
            x = layer(x, training=training)
        return self.fc_out(x)

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, nhead):
        super(DecoderLayer, self).__init__()
        self.self_attention = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model)
        self.feed_forward = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model * 4, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dropout2 = tf.keras.layers.Dropout(0.1)

    @tf.function
    def call(self, x, training=True):
        attn_output = self.self_attention(x, x, training=training)
        out1 = self.layer_norm1(x + self.dropout1(attn_output, training=training))
        ff_output = self.feed_forward(out1)
        out2 = self.layer_norm2(out1 + self.dropout2(ff_output, training=training))
        return out2

class TurnTakingDetector(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super(TurnTakingDetector, self).__init__()
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(d_model // 2, return_sequences=True))
        self.fc = tf.keras.layers.Dense(1, activation='sigmoid')

    @tf.function
    def call(self, x, training=True):
        x = self.lstm(x, training=training)
        return self.fc(x)

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.constant(pe, dtype=tf.float32)

    @tf.function
    def call(self, x):
        return x + self.pe[:tf.shape(x)[1], :]

class Vocoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_audio_tokens):
        super(Vocoder, self).__init__()
        self.prenet = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(d_model // 2, return_sequences=True))
        self.postnet = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=d_model, kernel_size=5, padding='same', activation='tanh'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Conv1D(filters=d_model, kernel_size=5, padding='same', activation='tanh'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Conv1D(filters=num_audio_tokens, kernel_size=5, padding='same')
        ])

    @tf.function
    def call(self, x, training=True):
        x = self.prenet(x)
        x = self.lstm(x, training=training)
        return self.postnet(x, training=training)

# Instantiate and test the model
model = LSLM(vocab_size=10000, d_model=512, nhead=8, num_layers=6, num_audio_tokens=1024)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Example of how to use the model
batch_size = 32
max_sequence_length = 100
speaking_input = tf.random.uniform((batch_size, max_sequence_length), maxval=10000, dtype=tf.int32)
listening_input = tf.random.normal((batch_size, max_sequence_length, 512))

# Forward pass
output = model([speaking_input, listening_input], training=True)
print(f"Output shape: {output.shape}")

# Generate
context = tf.random.uniform((1, 10), maxval=10000, dtype=tf.int32)
generated = model.generate(context)
print(f"Generated shape: {generated.shape}")