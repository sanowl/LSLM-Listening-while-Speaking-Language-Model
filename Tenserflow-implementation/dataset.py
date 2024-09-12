import tensorflow as tf
import tensorflow_io as tfio
from transformers import Wav2Vec2FeatureExtractor
import numpy as np

SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 10  # seconds

class LSLMDataset(tf.data.Dataset):
    """
    Custom dataset for the LSLM model.
    """
    def __new__(cls, num_samples=1000, max_length=MAX_AUDIO_LENGTH * SAMPLE_RATE):
        return tf.data.Dataset.from_generator(
            cls._generator,
            args=(num_samples, max_length),
            output_signature=(
                tf.TensorSpec(shape=(max_length,), dtype=tf.float32),
                tf.TensorSpec(shape=(max_length,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        )

    @staticmethod
    def _generator(num_samples, max_length):
        tokenizer = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        # Generate dummy data
        for i in range(num_samples):
            # Generate random audio samples and texts
            tts_sample = tf.random.normal((max_length,))
            text = f"Sample text {i}"
            tokenized_text = tokenizer(text, return_tensors="tf").input_values[0]

            # Simulate interruption and noise
            interruption = tf.random.normal((SAMPLE_RATE,))
            noise = tf.random.normal((max_length,))

            # Combine TTS sample with interruption and noise
            combined_audio = tts_sample + tf.pad(interruption, [[0, max_length - SAMPLE_RATE]]) + add_noise(tts_sample, snr_db=10)

            yield tts_sample, combined_audio, tf.cast(tokenized_text, tf.int32)

def add_noise(audio, snr_db):
    """
    Adds Gaussian noise to the audio signal at a specified SNR.

    Args:
        audio: Input audio signal.
        snr_db: Desired Signal-to-Noise Ratio in decibels.
    """
    signal_power = tf.reduce_mean(tf.square(audio))
    noise_power = signal_power / tf.pow(10.0, snr_db / 10.0)
    noise = tf.random.normal(tf.shape(audio), stddev=tf.sqrt(noise_power))
    return audio + noise

def create_tf_dataset(dataset, batch_size=32):
    """
    Converts the custom dataset into a batched TensorFlow dataset.

    Args:
        dataset: The LSLMDataset instance.
        batch_size: Number of samples per batch.
    """
    return dataset.padded_batch(
        batch_size,
        padded_shapes=(
            [MAX_AUDIO_LENGTH * SAMPLE_RATE],
            [MAX_AUDIO_LENGTH * SAMPLE_RATE],
            [None]
        ),
        padding_values=(0.0, 0.0, 0)
    )
