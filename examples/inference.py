#!/usr/bin/env python3
"""Example inference script for LSLM model."""

import torch
import torchaudio
from pathlib import Path
from transformers import BertTokenizer

from lslm.models.lslm import LSLMModel
from lslm.configs.default_config import LSLMConfig

def load_audio(audio_path, config):
    """Load and process audio file."""
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if sample_rate != config.audio_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, config.audio_sample_rate)
        waveform = resampler(waveform)
    
    # Convert to mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.audio_sample_rate,
        n_fft=config.hop_length * 4,
        hop_length=config.hop_length,
        n_mels=config.mel_bins
    )
    return mel_transform(waveform)

def main():
    # Load model
    checkpoint_path = "path/to/your/checkpoint.pt"
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint["config"]
    
    model = LSLMModel(config.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Process audio
    audio_path = "path/to/your/audio.wav"
    audio_features = load_audio(audio_path, config.data)
    audio_features = audio_features.unsqueeze(0).to(device)  # Add batch dimension
    
    # Generate text
    with torch.no_grad():
        generated_ids = model.generate(
            audio_features,
            max_length=50,
            temperature=0.7
        )
    
    # Decode text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main() 