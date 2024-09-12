import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import Wav2Vec2Model


class LSLM(nn.Module):
    """
    Listening-Speaking Language Model (LSLM) integrates listening and speaking encoders
    with a fusion module and a transformer-based decoder.
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

        # Components
        self.speaking_encoder = SpeakingEncoder(vocab_size, d_model, num_audio_tokens)
        self.listening_encoder = ListeningEncoder(d_model)
        self.fusion_module = FusionModule(d_model, fusion_type)
        self.decoder = Decoder(vocab_size, d_model, nhead, num_layers)
        self.turn_taking_detector = TurnTakingDetector(d_model)
        self.vocoder = Vocoder(d_model, num_audio_tokens)

        # Special IRQ token at the end of the vocabulary
        self.irq_token = vocab_size

    def forward(
        self,
        speaking_input: torch.Tensor,
        listening_input: torch.Tensor,
        is_training: bool = True
    ) -> torch.Tensor:
        """
        Forward pass of the LSLM model.

        Args:
            speaking_input (torch.Tensor): Input tensor for the speaking encoder.
            listening_input (torch.Tensor): Input tensor for the listening encoder.
            is_training (bool): Whether the model is in training mode.

        Returns:
            torch.Tensor: Output logits or IRQ token.
        """
        # Encode speaking and listening inputs
        speaking_features = self.speaking_encoder(speaking_input)
        listening_features = self.listening_encoder(listening_input)

        # Fuse features
        fused_features = self.fusion_module(speaking_features, listening_features)

        # Decode fused features
        output = self.decoder(fused_features)

        if not is_training:
            # Turn-taking detection
            turn_taking_prob = self.turn_taking_detector(listening_features)
            if turn_taking_prob.mean() > 0.5:
                # Return IRQ token if turn-taking condition is met
                batch_size = speaking_input.size(0)
                return torch.full((batch_size, 1), self.irq_token, dtype=torch.long, device=speaking_input.device)
        return output

    def generate(self, context: torch.Tensor, max_length: int = 1000) -> torch.Tensor:
        """
        Generates output tokens based on the given context.

        Args:
            context (torch.Tensor): Input context tensor.
            max_length (int): Maximum length of the generated sequence.

        Returns:
            torch.Tensor: Generated audio waveform from the vocoder.
        """
        generated_tokens = []
        listening_placeholder = torch.zeros(1, 1, self.d_model, device=context.device)
        for _ in range(max_length):
            output_logits = self.forward(context, listening_placeholder, is_training=False)
            # Get the predicted token (assuming output logits)
            predicted_token = output_logits.argmax(dim=-1)[:, -1]
            if predicted_token.item() == self.irq_token:
                break
            generated_tokens.append(predicted_token.unsqueeze(1))
            # Update context
            context = torch.cat([context, predicted_token.unsqueeze(1)], dim=1)
        if not generated_tokens:
            return torch.tensor([])  # Return empty tensor if nothing generated
        generated_sequence = torch.cat(generated_tokens, dim=1)
        # Convert tokens to audio using the vocoder
        generated_audio = self.vocoder(generated_sequence)
        return generated_audio


class SpeakingEncoder(nn.Module):
    """
    Encoder that handles both text and audio input, converting them into a shared feature space.
    """
    def __init__(self, vocab_size: int, d_model: int, num_audio_tokens: int):
        super(SpeakingEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, d_model)  # +1 for IRQ token
        self.positional_encoding = PositionalEncoding(d_model)
        self.audio_quantizer = AudioQuantizer(num_audio_tokens, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SpeakingEncoder.

        Args:
            x (torch.Tensor): Input tensor, either token IDs or raw audio features.

        Returns:
            torch.Tensor: Encoded features.
        """
        if x.dtype == torch.long:
            # Text input
            x = self.embedding(x)
        else:
            # Audio input
            x = self.audio_quantizer(x)
        x = self.positional_encoding(x)
        return x


class AudioQuantizer(nn.Module):
    """
    Converts raw audio into quantized tokens using a learned codebook.
    """
    def __init__(self, num_tokens: int, d_model: int):
        super(AudioQuantizer, self).__init__()
        self.codebook = nn.Parameter(torch.randn(num_tokens, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantizes the input audio features.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, feature_dim).

        Returns:
            torch.Tensor: Quantized embeddings.
        """
        # Flatten batch and sequence dimensions
        flat_x = x.view(-1, x.size(-1))
        # Compute distances to codebook vectors
        distances = torch.cdist(flat_x.unsqueeze(0), self.codebook.unsqueeze(0), p=2).squeeze(0)
        # Get indices of closest codebook vectors
        indices = distances.argmin(dim=1)
        # Get quantized embeddings
        quantized = self.codebook[indices]
        # Reshape back to original shape
        quantized = quantized.view(x.size())
        return quantized

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the indices of the closest codebook vectors.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Indices tensor.
        """
        flat_x = x.view(-1, x.size(-1))
        distances = torch.cdist(flat_x.unsqueeze(0), self.codebook.unsqueeze(0), p=2).squeeze(0)
        indices = distances.argmin(dim=1)
        indices = indices.view(x.size(0), x.size(1))
        return indices


class ListeningEncoder(nn.Module):
    """
    Encoder for converting raw audio inputs into features using a pre-trained Wav2Vec2 model.
    """
    def __init__(self, d_model: int):
        super(ListeningEncoder, self).__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.proj = nn.Linear(self.wav2vec.config.hidden_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ListeningEncoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Projected features of shape (batch_size, seq_len, d_model).
        """
        with torch.no_grad():
            x = self.wav2vec(x).last_hidden_state
        x = self.proj(x)
        return x


class FusionModule(nn.Module):
    """
    Fusion module that combines features from the speaking and listening encoders.
    Supports early, middle, and late fusion strategies.
    """
    def __init__(self, d_model: int, fusion_type: str = 'middle'):
        super(FusionModule, self).__init__()
        self.fusion_type = fusion_type.lower()
        self.d_model = d_model

        if self.fusion_type == 'early':
            self.fusion_layer = nn.Linear(2 * d_model, d_model)
        elif self.fusion_type == 'middle':
            self.fusion_layers = nn.ModuleList([nn.Linear(2 * d_model, d_model) for _ in range(6)])
        elif self.fusion_type == 'late':
            self.fusion_layer = nn.Linear(2 * d_model, d_model)
        else:
            raise ValueError(f"Invalid fusion_type '{fusion_type}'. Choose from 'early', 'middle', or 'late'.")

    def forward(self, speaking_features: torch.Tensor, listening_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FusionModule.

        Args:
            speaking_features (torch.Tensor): Tensor from the speaking encoder.
            listening_features (torch.Tensor): Tensor from the listening encoder.

        Returns:
            torch.Tensor: Fused features tensor.
        """
        if self.fusion_type == 'early':
            fused = self.fusion_layer(torch.cat([speaking_features, listening_features], dim=-1))
        elif self.fusion_type == 'middle':
            fused = speaking_features
            for layer in self.fusion_layers:
                fused = layer(torch.cat([fused, listening_features], dim=-1))
        elif self.fusion_type == 'late':
            fused = self.fusion_layer(torch.cat([speaking_features, listening_features], dim=-1))
        return fused


class Decoder(nn.Module):
    """
    Transformer-based decoder that generates text from fused features.
    """
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int):
        super(Decoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size + 1)  # +1 for IRQ token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, seq_len, vocab_size + 1).
        """
        # Prepare masks for transformer decoder
        seq_len = x.size(1)
        tgt_mask = self._generate_square_subsequent_mask(seq_len).to(x.device)
        x = x.transpose(0, 1)  # Transformer expects (seq_len, batch_size, d_model)
        x = self.transformer_decoder(x, x, tgt_mask=tgt_mask)
        x = x.transpose(0, 1)  # Back to (batch_size, seq_len, d_model)
        logits = self.fc_out(x)
        return logits

    @staticmethod
    def _generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        """
        Generates an upper-triangular matrix of -inf, with zeros on diag.

        Args:
            sz (int): Size of the mask.

        Returns:
            torch.Tensor: The mask tensor.
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


class TurnTakingDetector(nn.Module):
    """
    Detects turn-taking based on listening features.
    """
    def __init__(self, d_model: int):
        super(TurnTakingDetector, self).__init__()
        self.lstm = nn.LSTM(d_model, d_model // 2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TurnTakingDetector.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Probability of turn-taking.
        """
        x, _ = self.lstm(x)
        logits = self.fc(x)  # (batch_size, seq_len, 1)
        probs = torch.sigmoid(logits)
        # Aggregate over sequence length
        turn_taking_prob = probs.mean(dim=1)  # (batch_size, 1)
        return turn_taking_prob


class PositionalEncoding(nn.Module):
    """
    Adds positional information to the encoded features.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0, d_model, 2).float() * (math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # dim 2i
        pe[:, 1::2] = torch.cos(position * div_term)  # dim 2i+1
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x


class Vocoder(nn.Module):
    """
    Converts token sequences into audio waveforms.
    """
    def __init__(self, d_model: int, num_audio_tokens: int):
        super(Vocoder, self).__init__()
        self.num_audio_tokens = num_audio_tokens
        self.prenet = nn.Sequential(
            nn.Linear(num_audio_tokens, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.lstm = nn.LSTM(d_model, d_model // 2, num_layers=2, batch_first=True, bidirectional=True)
        self.postnet = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(d_model, 1, kernel_size=5, padding=2)  # Output is 1-dimensional waveform
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vocoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Generated audio waveform tensor of shape (batch_size, seq_len).
        """
        x = F.one_hot(x, num_classes=self.num_audio_tokens).float()  # Convert tokens to one-hot
        x = self.prenet(x)
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)  # For Conv1d: (batch_size, d_model, seq_len)
        x = self.postnet(x)
        x = x.squeeze(1)  # Remove channel dimension
        return x
