"""Dataset class for LSLM model."""

import torch
from torch.utils.data import Dataset
import torchaudio
import numpy as np
from pathlib import Path
import json

class LSLMDataset(Dataset):
    def __init__(self, data_dir, config, split="train"):
        """
        Args:
            data_dir: Path to data directory
            config: Data configuration
            split: train, val, or test
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.split = split
        
        # Load metadata
        with open(self.data_dir / f"{split}_metadata.json", "r") as f:
            self.metadata = json.load(f)
            
        # Setup audio processing
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.audio_sample_rate,
            n_fft=config.hop_length * 4,
            hop_length=config.hop_length,
            n_mels=config.mel_bins
        )
        
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load and process audio
        audio_path = self.data_dir / "audio" / item["audio_file"]
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if necessary
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if necessary
        if sample_rate != self.config.audio_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.config.audio_sample_rate)
            waveform = resampler(waveform)
            
        # Convert to mel spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Load text
        text_ids = torch.tensor(item["text_ids"], dtype=torch.long)
        
        # Create attention mask
        attention_mask = torch.ones_like(text_ids, dtype=torch.bool)
        
        return {
            "audio_features": mel_spec,
            "text_input_ids": text_ids,
            "attention_mask": attention_mask,
            "metadata": {
                "audio_file": item["audio_file"],
                "text": item["text"]
            }
        }
        
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for DataLoader."""
        # Get max lengths
        max_audio_len = max(x["audio_features"].size(-1) for x in batch)
        max_text_len = max(x["text_input_ids"].size(0) for x in batch)
        
        # Prepare tensors
        audio_features = torch.zeros(len(batch), batch[0]["audio_features"].size(0), max_audio_len)
        text_input_ids = torch.zeros(len(batch), max_text_len, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_text_len, dtype=torch.bool)
        
        # Fill tensors
        for i, item in enumerate(batch):
            # Audio
            audio_len = item["audio_features"].size(-1)
            audio_features[i, :, :audio_len] = item["audio_features"]
            
            # Text
            text_len = item["text_input_ids"].size(0)
            text_input_ids[i, :text_len] = item["text_input_ids"]
            attention_mask[i, :text_len] = item["attention_mask"]
            
        return {
            "audio_features": audio_features,
            "text_input_ids": text_input_ids,
            "attention_mask": attention_mask,
            "metadata": [x["metadata"] for x in batch]
        } 