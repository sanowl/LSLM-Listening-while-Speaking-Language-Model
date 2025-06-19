"""data augmentation for LSLM training."""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
import random
from typing import Tuple, Optional, List
from torch_audiomentations import (
    Compose, 
    AddColoredNoise, 
    TimeStretch, 
    PitchShift,
    Gain,
    PolarityInversion
)


class AudioAugmentation(nn.Module):
    """Professional audio augmentation pipeline."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        time_masking: bool = True,
        freq_masking: bool = True,
        noise_prob: float = 0.3,
        speed_perturbation: bool = True,
        pitch_shift: bool = True,
        gain_augment: bool = True
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.time_masking = time_masking
        self.freq_masking = freq_masking
        
        # Audio augmentations
        augmentations = []
        
        if noise_prob > 0:
            augmentations.append(
                AddColoredNoise(
                    min_snr_in_db=10.0,
                    max_snr_in_db=30.0,
                    min_f_decay=-2.0,
                    max_f_decay=2.0,
                    p=noise_prob
                )
            )
            
        if speed_perturbation:
            augmentations.append(
                TimeStretch(
                    min_rate=0.8,
                    max_rate=1.2,
                    p=0.3
                )
            )
            
        if pitch_shift:
            augmentations.append(
                PitchShift(
                    min_transpose_semitones=-4.0,
                    max_transpose_semitones=4.0,
                    p=0.3
                )
            )
            
        if gain_augment:
            augmentations.append(
                Gain(
                    min_gain_in_db=-12.0,
                    max_gain_in_db=12.0,
                    p=0.3
                )
            )
            
        augmentations.append(
            PolarityInversion(p=0.1)
        )
        
        self.waveform_augment = Compose(
            transforms=augmentations,
            p=0.8
        )
        
        # Spectrogram masking
        if time_masking:
            self.time_mask = torchaudio.transforms.TimeMasking(
                time_mask_param=40,
                p=0.3
            )
        else:
            self.time_mask = None
            
        if freq_masking:
            self.freq_mask = torchaudio.transforms.FrequencyMasking(
                freq_mask_param=15,
                p=0.3
            )
        else:
            self.freq_mask = None
            
    def forward(
        self, 
        waveform: torch.Tensor,
        spectrogram: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply augmentation to waveform and/or spectrogram.
        
        Args:
            waveform: Audio waveform (batch_size, 1, time)
            spectrogram: Mel spectrogram (batch_size, mel_bins, time)
            
        Returns:
            Tuple of (augmented_waveform, augmented_spectrogram)
        """
        # Augment waveform
        if self.training:
            waveform = self.waveform_augment(
                waveform, 
                sample_rate=self.sample_rate
            )
            
        # Augment spectrogram
        if spectrogram is not None and self.training:
            if self.time_mask is not None:
                spectrogram = self.time_mask(spectrogram)
            if self.freq_mask is not None:
                spectrogram = self.freq_mask(spectrogram)
                
        return waveform, spectrogram


class TextAugmentation:
    """Text augmentation strategies."""
    
    def __init__(
        self,
        token_dropout_prob: float = 0.1,
        token_replacement_prob: float = 0.05,
        sequence_shuffle_prob: float = 0.1,
        vocab_size: int = 30522
    ):
        self.token_dropout_prob = token_dropout_prob
        self.token_replacement_prob = token_replacement_prob
        self.sequence_shuffle_prob = sequence_shuffle_prob
        self.vocab_size = vocab_size
        
    def token_dropout(self, input_ids: torch.Tensor, mask_token_id: int = 103) -> torch.Tensor:
        """Randomly mask tokens."""
        if random.random() > self.token_dropout_prob:
            return input_ids
            
        mask = torch.rand_like(input_ids.float()) < 0.15
        # Don't mask special tokens (first and last)
        mask[:, 0] = False
        mask[:, -1] = False
        
        input_ids = input_ids.clone()
        input_ids[mask] = mask_token_id
        
        return input_ids
        
    def token_replacement(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Randomly replace tokens with random vocabulary tokens."""
        if random.random() > self.token_replacement_prob:
            return input_ids
            
        mask = torch.rand_like(input_ids.float()) < 0.05
        # Don't replace special tokens
        mask[:, 0] = False
        mask[:, -1] = False
        
        input_ids = input_ids.clone()
        random_tokens = torch.randint_like(input_ids, low=0, high=self.vocab_size)
        input_ids[mask] = random_tokens[mask]
        
        return input_ids
        
    def sequence_shuffle(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Randomly shuffle subsequences."""
        if random.random() > self.sequence_shuffle_prob:
            return input_ids
            
        batch_size, seq_len = input_ids.shape
        input_ids = input_ids.clone()
        
        for i in range(batch_size):
            # Choose random subsequence to shuffle
            start = random.randint(1, seq_len - 3)  # Avoid special tokens
            end = random.randint(start + 1, seq_len - 1)
            
            # Shuffle the subsequence
            subsequence = input_ids[i, start:end]
            shuffled_indices = torch.randperm(len(subsequence))
            input_ids[i, start:end] = subsequence[shuffled_indices]
            
        return input_ids
        
    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply all text augmentations."""
        input_ids = self.token_dropout(input_ids)
        input_ids = self.token_replacement(input_ids)
        input_ids = self.sequence_shuffle(input_ids)
        return input_ids


class MixedModalityAugmentation:
    """Advanced cross-modal augmentation strategies."""
    
    def __init__(
        self,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        modal_dropout_prob: float = 0.1
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.modal_dropout_prob = modal_dropout_prob
        
    def mixup(
        self, 
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply MixUp augmentation."""
        if random.random() > 0.5:
            return audio_features, text_features, labels, 1.0
            
        batch_size = audio_features.size(0)
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        # Random permutation
        index = torch.randperm(batch_size)
        
        # Mix features
        mixed_audio = lam * audio_features + (1 - lam) * audio_features[index]
        mixed_text = lam * text_features + (1 - lam) * text_features[index]
        mixed_labels = lam * labels + (1 - lam) * labels[index]
        
        return mixed_audio, mixed_text, mixed_labels, lam
        
    def modal_dropout(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly drop one modality."""
        if random.random() > self.modal_dropout_prob:
            return audio_features, text_features
            
        if random.random() > 0.5:
            # Drop audio
            audio_features = torch.zeros_like(audio_features)
        else:
            # Drop text
            text_features = torch.zeros_like(text_features)
            
        return audio_features, text_features
        
    def temporal_alignment_noise(
        self,
        audio_features: torch.Tensor,
        max_shift: int = 10
    ) -> torch.Tensor:
        """Add temporal misalignment between modalities."""
        if random.random() > 0.3:
            return audio_features
            
        batch_size, channels, time_steps = audio_features.shape
        
        for i in range(batch_size):
            shift = random.randint(-max_shift, max_shift)
            if shift > 0:
                # Shift right (pad left)
                audio_features[i, :, shift:] = audio_features[i, :, :-shift]
                audio_features[i, :, :shift] = 0
            elif shift < 0:
                # Shift left (pad right)
                audio_features[i, :, :shift] = audio_features[i, :, -shift:]
                audio_features[i, :, shift:] = 0
                
        return audio_features


class AdvancedDataAugmentation(nn.Module):
    """Complete augmentation pipeline for LSLM training."""
    
    def __init__(
        self,
        audio_config: dict,
        text_config: dict,
        mixed_config: dict
    ):
        super().__init__()
        
        self.audio_augment = AudioAugmentation(**audio_config)
        self.text_augment = TextAugmentation(**text_config)
        self.mixed_augment = MixedModalityAugmentation(**mixed_config)
        
    def forward(
        self,
        waveform: torch.Tensor,
        spectrogram: torch.Tensor,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Apply comprehensive augmentation pipeline.
        
        Args:
            waveform: Audio waveform
            spectrogram: Mel spectrogram
            input_ids: Text token IDs
            labels: Training labels
            
        Returns:
            Dictionary with augmented features
        """
        # Audio augmentation
        aug_waveform, aug_spectrogram = self.audio_augment(waveform, spectrogram)
        
        # Text augmentation
        aug_input_ids = self.text_augment(input_ids)
        
        # Mixed modality augmentation
        if labels is not None and self.training:
            aug_audio, aug_text, aug_labels, mixup_lam = self.mixed_augment.mixup(
                aug_spectrogram, aug_input_ids.float(), labels
            )
            aug_audio, aug_text = self.mixed_augment.modal_dropout(aug_audio, aug_text)
            aug_audio = self.mixed_augment.temporal_alignment_noise(aug_audio)
        else:
            aug_audio = aug_spectrogram
            aug_text = aug_input_ids
            aug_labels = labels
            mixup_lam = 1.0
            
        return {
            'waveform': aug_waveform,
            'spectrogram': aug_audio,
            'input_ids': aug_text.long() if aug_text.dtype != torch.long else aug_text,
            'labels': aug_labels,
            'mixup_lambda': mixup_lam
        } 