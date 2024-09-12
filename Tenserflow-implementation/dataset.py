import tensorflow as tf
from transformers import Wav2Vec2Tokenizer
import numpy as np

SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 10  # seconds
MAX_AUDIO_SAMPLES = SAMPLE_RATE * MAX_AUDIO_LENGTH

class LSLMDataset:
    """
    Dataset class for the LSLM model.
    Generates synthetic data for training and evaluation.
    """
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random audio samples and texts
        tts_sample = tf.random.normal((MAX_AUDIO_SAMPLES,))
        text = f"Sample text {idx}"
        tokenized_text = self.tokenizer(text, return_tensors="tf").input_ids[0]

        # Simulate interruption and noise
        interruption_length = tf.random.uniform((), minval=1, maxval=SAMPLE_RATE, dtype=tf.int32)
        interruption = tf.random.normal((interruption_length,))
        noise = tf.random.normal((MAX_AUDIO_SAMPLES,))

        # Combine TTS sample with interruption and noise
        interruption_padded = tf.pad(interruption, [[0, MAX_AUDIO_SAMPLES - interruption_length]])
        combined_audio = tts_sample + interruption_padded + add_noise(tts_sample, snr_db=10)

        return tts_sample, combined_audio, tf.cast(tokenized_text, tf.int32)

def add_noise(audio, snr_db):
    """
    Adds Gaussian noise to the audio signal at a specified SNR.

    Args:
        audio: Input audio signal.
        snr_db: Desired Signal-to-Noise Ratio in decibels.

    Returns:
        Noisy audio signal.
    """
    signal_power = tf.reduce_mean(tf.square(audio))
    noise_power = signal_power / tf.pow(10.0, snr_db / 10.0)
    noise = tf.random.normal(tf.shape(audio), stddev=tf.sqrt(noise_power))
    return audio + noise

def create_tf_dataset(dataset, batch_size=32, shuffle=True):
    """
    Converts the custom dataset into a batched TensorFlow dataset.

    Args:
        dataset: The LSLMDataset instance.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the dataset.

    Returns:
        A tf.data.Dataset object.
    """
    def generator():
        for i in range(len(dataset)):
            yield dataset[i]

    output_signature = (
        tf.TensorSpec(shape=(MAX_AUDIO_SAMPLES,), dtype=tf.float32),
        tf.TensorSpec(shape=(MAX_AUDIO_SAMPLES,), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )

    tf_dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )

    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=1000)

    tf_dataset = tf_dataset.padded_batch(
        batch_size,
        padded_shapes=(
            [MAX_AUDIO_SAMPLES],
            [MAX_AUDIO_SAMPLES],
            [None]
        ),
        padding_values=(0.0, 0.0, 0)
    ).prefetch(tf.data.AUTOTUNE)

    return tf_dataset
