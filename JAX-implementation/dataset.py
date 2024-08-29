import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from flax import linen as nn
from flax.training import common_utils
from transformers import Wav2Vec2FeatureExtractor

SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 10  # seconds

def add_noise(audio, snr_db, key):
    signal_power = jnp.linalg.norm(audio)
    noise = random.normal(key, shape=audio.shape)
    noise_power = jnp.linalg.norm(noise)
    snr = 10**(snr_db/10)
    scale = snr * noise_power / signal_power
    noisy_audio = (scale * audio + noise) / (scale + 1)
    return noisy_audio

class LSLMDataset:
    def __init__(self, num_samples=1000, max_length=MAX_AUDIO_LENGTH * SAMPLE_RATE, key=None):
        self.num_samples = num_samples
        self.max_length = max_length
        self.tokenizer = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")

        if key is None:
            key = random.PRNGKey(0)

        self.key = key
        self.tts_data = [(random.normal(key, (max_length,)), f"Sample text {i}") for i in range(num_samples)]
        self.interruption_data = [random.normal(key, (SAMPLE_RATE,)) for _ in range(100)]  # 100 different interruptions
        self.noise_data = [random.normal(key, (max_length,)) for _ in range(10)]  # 10 different noise patterns

    def __getitem__(self, idx):
        key, subkey = random.split(self.key)
        tts_sample, text = self.tts_data[idx]
        interruption_idx = random.randint(key, (1,), 0, len(self.interruption_data))[0]
        noise_idx = random.randint(subkey, (1,), 0, len(self.noise_data))[0]
        
        interruption = self.interruption_data[interruption_idx]
        noise = self.noise_data[noise_idx]

        combined_audio = tts_sample + jnp.pad(interruption, (0, self.max_length - interruption.shape[0])) + add_noise(tts_sample, snr_db=10, key=subkey)
        
        # Tokenize text
        tokenized_text = self.tokenizer(text, return_tensors="jax").input_values.squeeze(0)
        
        return tts_sample, combined_audio, tokenized_text

    def __len__(self):
        return self.num_samples

def collate_fn(batch):
    tts_samples, combined_audios, tokenized_texts = zip(*batch)
    
    # Convert to JAX arrays and pad sequences
    tts_samples = jnp.array(common_utils.pad_sequences(tts_samples, pad_value=0))
    combined_audios = jnp.array(common_utils.pad_sequences(combined_audios, pad_value=0))
    
    # Pad tokenized texts
    tokenized_texts = jnp.array(common_utils.pad_sequences(tokenized_texts, pad_value=0))
    
    return tts_samples, combined_audios, tokenized_texts

# Example usage:
dataset = LSLMDataset()
batch = [dataset[i] for i in range(4)]
collated_batch = collate_fn(batch)

print(collated_batch)
