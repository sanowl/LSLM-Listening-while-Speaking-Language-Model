import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

class LSLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_audio_tokens):
        super().__init__()
        self.speaking_encoder = SpeakingEncoder(vocab_size, d_model, num_audio_tokens)
        self.listening_encoder = ListeningEncoder(d_model)
        self.fusion_module = FusionModule(d_model)
        self.decoder = Decoder(vocab_size, d_model, nhead, num_layers)
        self.irq_token = vocab_size  # Assume IRQ token is the last token
        self.turn_taking_detector = TurnTakingDetector(d_model)
        self.vocoder = Vocoder(d_model, num_audio_tokens)

    def forward(self, speaking_input, listening_input, is_training=True):
        speaking_features = self.speaking_encoder(speaking_input)
        listening_features = self.listening_encoder(listening_input)
        fused_features = self.fusion_module(speaking_features, listening_features)
        output = self.decoder(fused_features)
        
        if not is_training:
            turn_taking = self.turn_taking_detector(listening_features)
            if turn_taking:
                return self.irq_token
        
        return output

    def generate(self, context, max_length=1000):
        generated = []
        for _ in range(max_length):
            output = self.forward(context, torch.zeros(1, 1, self.listening_encoder.d_model), is_training=False)
            if output == self.irq_token:
                break
            generated.append(output)
            context = torch.cat([context, output.unsqueeze(0)], dim=1)
        return self.vocoder(torch.cat(generated, dim=0))

class SpeakingEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_audio_tokens):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, d_model)  # +1 for IRQ token
        self.positional_encoding = PositionalEncoding(d_model)
        self.audio_quantizer = AudioQuantizer(num_audio_tokens, d_model)

    def forward(self, x):
        if isinstance(x, torch.LongTensor):  # Text input
            x = self.embedding(x)
        else:  # Audio input
            x = self.audio_quantizer(x)
        x = self.positional_encoding(x)
        return x

class AudioQuantizer(nn.Module):
    def __init__(self, num_tokens, d_model):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, d_model)
        self.codebook = nn.Parameter(torch.randn(num_tokens, d_model))

    def forward(self, x):
        distances = torch.cdist(x, self.codebook, p=2)
        indices = torch.argmin(distances, dim=-1)
        return self.embedding(indices)

    def quantize(self, x):
        distances = torch.cdist(x, self.codebook, p=2)
        indices = torch.argmin(distances, dim=-1)
        return indices

class ListeningEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        self.proj = nn.Linear(self.wav2vec.config.hidden_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        with torch.no_grad():
            inputs = self.feature_extractor(x, return_tensors="pt", padding=True)
            x = self.wav2vec(**inputs).last_hidden_state
        return self.proj(x)

class FusionModule(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.early_fusion = nn.Linear(2 * d_model, d_model)
        self.middle_fusion = nn.ModuleList([nn.Linear(2 * d_model, d_model) for _ in range(6)])
        self.late_fusion = nn.Linear(2 * d_model, d_model)
        self.fusion_type = 'middle'  # Can be 'early', 'middle', or 'late'

    def forward(self, speaking_features, listening_features):
        if self.fusion_type == 'early':
            return self.early_fusion(torch.cat([speaking_features, listening_features], dim=-1))
        elif self.fusion_type == 'middle':
            fused = speaking_features
            for layer in self.middle_fusion:
                fused = layer(torch.cat([fused, listening_features], dim=-1))
            return fused
        elif self.fusion_type == 'late':
            return self.late_fusion(torch.cat([speaking_features, listening_features], dim=-1))

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead),
            num_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size + 1)  # +1 for IRQ token

    def forward(self, x):
        x = self.transformer_decoder(x, x)  # Self-attention
        return self.fc_out(x)

class TurnTakingDetector(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.lstm = nn.LSTM(d_model, d_model // 2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return torch.sigmoid(x).squeeze(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class Vocoder(nn.Module):
    def __init__(self, d_model, num_audio_tokens):
        super().__init__()
        self.prenet = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.lstm = nn.LSTM(d_model, d_model // 2, 2, batch_first=True)
        self.postnet = nn.Sequential(
            nn.Conv1d(d_model // 2, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(d_model, num_audio_tokens, kernel_size=5, padding=2)
        )

    def forward(self, x):
        x = self.prenet(x)
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)
        x = self.postnet(x)
        return x.transpose(1, 2)
