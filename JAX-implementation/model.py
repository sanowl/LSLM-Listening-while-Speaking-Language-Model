import jax
import jax.numpy as jnp
import flax.linen as nn
import math
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import numpy as np

class LSLM(nn.Module):
    vocab_size: int
    d_model: int
    nhead: int
    num_layers: int
    num_audio_tokens: int

    def setup(self):
        self.speaking_encoder = SpeakingEncoder(self.vocab_size, self.d_model, self.num_audio_tokens)
        self.listening_encoder = ListeningEncoder(self.d_model)
        self.fusion_module = FusionModule(self.d_model)
        self.decoder = Decoder(self.vocab_size, self.d_model, self.nhead, self.num_layers)
        self.irq_token = self.vocab_size  # Assume IRQ token is the last token
        self.turn_taking_detector = TurnTakingDetector(self.d_model)
        self.vocoder = Vocoder(self.d_model, self.num_audio_tokens)

    def __call__(self, speaking_input, listening_input, is_training=True):
        speaking_features = self.speaking_encoder(speaking_input)
        listening_features = self.listening_encoder(listening_input)
        fused_features = self.fusion_module(speaking_features, listening_features)
        output = self.decoder(fused_features)

        if not is_training:
            turn_taking = self.turn_taking_detector(listening_features)
            if turn_taking > 0.5:
                return self.irq_token

        return output

    def generate(self, context, max_length=1000):
        generated = []
        for _ in range(max_length):
            output = self(context, jnp.zeros((1, 1, self.listening_encoder.d_model)), is_training=False)
            if output == self.irq_token:
                break
            generated.append(output)
            context = jnp.concatenate([context, output.reshape(1, 1, -1)], axis=1)
        return self.vocoder(jnp.concatenate(generated, axis=0))

class SpeakingEncoder(nn.Module):
    vocab_size: int
    d_model: int
    num_audio_tokens: int

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size + 1, self.d_model)  # +1 for IRQ token
        self.positional_encoding = PositionalEncoding(self.d_model)
        self.audio_quantizer = AudioQuantizer(self.num_audio_tokens, self.d_model)

    def __call__(self, x):
        if x.dtype == jnp.int32:  # Text input
            x = self.embedding(x)
        else:  # Audio input
            x = self.audio_quantizer(x)
        x = self.positional_encoding(x)
        return x

class AudioQuantizer(nn.Module):
    num_tokens: int
    d_model: int

    def setup(self):
        self.embedding = nn.Embed(self.num_tokens, self.d_model)
        self.codebook = self.param("codebook", nn.initializers.normal(), (self.num_tokens, self.d_model))

    def __call__(self, x):
        distances = jnp.linalg.norm(x[:, None] - self.codebook[None, :], axis=-1)
        indices = jnp.argmin(distances, axis=-1)
        return self.embedding(indices)

    def quantize(self, x):
        distances = jnp.linalg.norm(x[:, None] - self.codebook[None, :], axis=-1)
        indices = jnp.argmin(distances, axis=-1)
        return indices

class ListeningEncoder(nn.Module):
    d_model: int

    def setup(self):
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        self.proj = nn.Dense(self.d_model)

    def __call__(self, x):
        inputs = self.feature_extractor(x, return_tensors="jax", padding=True)
        hidden_states = self.wav2vec(**inputs).last_hidden_state
        return self.proj(hidden_states)

class FusionModule(nn.Module):
    d_model: int
    fusion_type: str = 'middle'  # Can be 'early', 'middle', or 'late'

    def setup(self):
        self.early_fusion = nn.Dense(self.d_model)
        self.middle_fusion = [nn.Dense(self.d_model) for _ in range(6)]
        self.late_fusion = nn.Dense(self.d_model)

    def __call__(self, speaking_features, listening_features):
        if self.fusion_type == 'early':
            return self.early_fusion(jnp.concatenate([speaking_features, listening_features], axis=-1))
        elif self.fusion_type == 'middle':
            fused = speaking_features
            for layer in self.middle_fusion:
                fused = layer(jnp.concatenate([fused, listening_features], axis=-1))
            return fused
        elif self.fusion_type == 'late':
            return self.late_fusion(jnp.concatenate([speaking_features, listening_features], axis=-1))

class Decoder(nn.Module):
    vocab_size: int
    d_model: int
    nhead: int
    num_layers: int

    def setup(self):
        self.transformer_decoder = nn.Transformer(
            num_layers=self.num_layers,
            num_heads=self.nhead,
            dim_model=self.d_model,
            mlp_dim=self.d_model * 4,
        )
        self.fc_out = nn.Dense(self.vocab_size + 1)  # +1 for IRQ token

    def __call__(self, x):
        x = self.transformer_decoder(x)
        return self.fc_out(x)

class TurnTakingDetector(nn.Module):
    d_model: int

    def setup(self):
        self.lstm = nn.LSTM(self.d_model // 2, bidirectional=True)
        self.fc = nn.Dense(1)

    def __call__(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return nn.sigmoid(x).squeeze(-1)

class PositionalEncoding(nn.Module):
    d_model: int
    max_len: int = 5000

    def setup(self):
        position = jnp.arange(self.max_len).reshape(-1, 1)
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe = jnp.zeros((self.max_len, self.d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pe = pe

    def __call__(self, x):
        return x + self.pe[:x.shape[1], :]

class Vocoder(nn.Module):
    d_model: int
    num_audio_tokens: int

    def setup(self):
        self.prenet = nn.Sequential([
            nn.Dense(self.d_model),
            nn.relu,
            nn.Dense(self.d_model)
        ])
        self.lstm = nn.LSTM(self.d_model // 2, bidirectional=True)
        self.postnet = nn.Sequential([
            nn.Conv(features=self.d_model, kernel_size=(5,)),
            nn.BatchNorm(),
            nn.tanh,
            nn.Dropout(0.5),
            nn.Conv(features=self.d_model, kernel_size=(5,)),
            nn.BatchNorm(),
            nn.tanh,
            nn.Dropout(0.5),
            nn.Conv(features=self.num_audio_tokens, kernel_size=(5,))
        ])

    def __call__(self, x):
        x = self.prenet(x)
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)
        x = self.postnet(x)
        return x.transpose(1, 2)
