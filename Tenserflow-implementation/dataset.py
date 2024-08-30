import tensorflow as tf
import tensorflow_io as tfio
from transformers import Wav2Vec2FeatureExtractor

SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 10  # seconds

def add_noise(audio, snr_db):
    signal_power = tf.norm(audio, ord=2)
    noise = tf.random.normal(tf.shape(audio), dtype=audio.dtype)
    noise_power = tf.norm(noise, ord=2)
    snr = tf.pow(10.0, snr_db / 10.0)
    scale = snr * noise_power / signal_power
    noisy_audio = (scale * audio + noise) / (scale + 1)
    return noisy_audio

class LSLMDataset(tf.data.Dataset):
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
        tts_data = [(tf.random.normal((max_length,)), f"Sample text {i}") for i in range(num_samples)]
        interruption_data = [tf.random.normal((SAMPLE_RATE,)) for _ in range(100)]
        noise_data = [tf.random.normal((max_length,)) for _ in range(10)]

        for tts_sample, text in tts_data:
            interruption = interruption_data[tf.random.uniform((), minval=0, maxval=len(interruption_data), dtype=tf.int32)]
            noise = noise_data[tf.random.uniform((), minval=0, maxval=len(noise_data), dtype=tf.int32)]

            # Combine TTS sample with interruption and noise
            padded_interruption = tf.pad(interruption, [[0, max_length - tf.shape(interruption)[0]]])
            combined_audio = tts_sample + padded_interruption + add_noise(tts_sample, snr_db=10)

            # Tokenize text
            tokenized_text = tokenizer(text, return_tensors="pt").input_values.squeeze(0).numpy()

            yield tts_sample, combined_audio, tokenized_text

def create_tf_dataset(dataset, batch_size=32):
    return dataset.padded_batch(
        batch_size,
        padded_shapes=(
            [MAX_AUDIO_LENGTH * SAMPLE_RATE],
            [MAX_AUDIO_LENGTH * SAMPLE_RATE],
            [None]
        ),
        padding_values=(0.0, 0.0, 0)
    )