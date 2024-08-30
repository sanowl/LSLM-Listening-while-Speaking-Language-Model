import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, MultiHeadAttention, LayerNormalization
from tensorflow.data import Dataset
import numpy as np
import tensorflow_io as tfio

from model import LSLM
from dataset import LSLMDataset, create_tf_dataset
from train import train, evaluate
from visualization import visualize_attention, visualize_quantization
from tests import command_based_fdm_test, voice_based_fdm_test, analyze_turn_taking, generate_speech

VOCAB_SIZE = 10000
D_MODEL = 512
NHEAD = 8
NUM_LAYERS = 6
NUM_AUDIO_TOKENS = 1024
SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 10  # seconds

def main():
    # Check for GPU
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # Create dataset and split into train, validation, and test
    full_dataset = LSLMDataset(num_samples=10000)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset = create_tf_dataset(full_dataset[:train_size])
    val_dataset = create_tf_dataset(full_dataset[train_size:train_size+val_size])
    test_dataset = create_tf_dataset(full_dataset[train_size+val_size:])

    # Initialize model
    model = LSLM(VOCAB_SIZE, D_MODEL, NHEAD, NUM_LAYERS, NUM_AUDIO_TOKENS)

    # Initialize optimizer and loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Training
    num_epochs = 10
    train(model, train_dataset, optimizer, loss_fn, num_epochs)

    # Evaluation
    val_loss = evaluate(model, val_dataset, loss_fn)
    print(f"Validation Loss: {val_loss:.4f}")

    # Test
    test_loss = evaluate(model, test_dataset, loss_fn)
    print(f"Test Loss: {test_loss:.4f}")

    # Ablation studies
    fusion_types = ['early', 'middle', 'late']
    for fusion_type in fusion_types:
        model.fusion_module.fusion_type = fusion_type
        test_loss = evaluate(model, test_dataset, loss_fn)
        print(f"Fusion type: {fusion_type}, Test Loss: {test_loss:.4f}")

    # Visualization
    sample_input = next(iter(test_dataset))
    visualize_attention(model, sample_input)

    # Command-based FDM test
    command_test_data = [(tf.random.normal((MAX_AUDIO_LENGTH * SAMPLE_RATE,)), "Honey") for _ in range(100)]
    command = model.speaking_encoder.tokenizer("Honey", return_tensors="tf").input_values
    command_based_fdm_test(model, command_test_data, command)

    # Voice-based FDM test
    voice_test_data = [(tf.random.normal((MAX_AUDIO_LENGTH * SAMPLE_RATE,)), tf.random.normal((SAMPLE_RATE,))) for _ in range(100)]
    voice_based_fdm_test(model, voice_test_data)

    # Turn-taking analysis
    turn_taking_test_data = [(tf.random.normal((MAX_AUDIO_LENGTH * SAMPLE_RATE,)), tf.random.uniform((), minval=0, maxval=2, dtype=tf.int32)) for _ in range(100)]
    analyze_turn_taking(model, turn_taking_test_data)

    # Generate speech from text
    test_text = "Hello, this is a test of the LSLM model."
    generated_speech = generate_speech(model, test_text)
    tfio.audio.write_wav('generated_speech.wav', generated_speech, SAMPLE_RATE)

    # Visualize audio quantization
    test_audio = next(iter(test_dataset))[0][0]
    visualize_quantization(model, test_audio)

    # Save model
    model.save_weights('lslm_model')

    print("Training, evaluation, and testing complete. Model saved.")

if __name__ == "__main__":
    main()