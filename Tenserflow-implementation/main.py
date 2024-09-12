import tensorflow as tf
from transformers import Wav2Vec2Tokenizer
from model import LSLM
from dataset import LSLMDataset, create_tf_dataset
from train import train, evaluate

VOCAB_SIZE = 10000
D_MODEL = 512
NHEAD = 8
NUM_LAYERS = 6
NUM_AUDIO_TOKENS = 1024
SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 10  # seconds

def main():
    # Check for GPU
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Create dataset and split into train, validation, and test
    full_dataset = LSLMDataset(num_samples=10000)
    full_dataset = list(full_dataset)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))

    train_dataset = create_tf_dataset(tf.data.Dataset.from_tensor_slices(full_dataset[:train_size]))
    val_dataset = create_tf_dataset(tf.data.Dataset.from_tensor_slices(full_dataset[train_size:train_size+val_size]))
    test_dataset = create_tf_dataset(tf.data.Dataset.from_tensor_slices(full_dataset[train_size+val_size:]))

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

    # Testing
    test_loss = evaluate(model, test_dataset, loss_fn)
    print(f"Test Loss: {test_loss:.4f}")

    # Save model
    model.save_weights('lslm_model')

    print("Training, evaluation, and testing complete. Model saved.")

if __name__ == "__main__":
    main()
