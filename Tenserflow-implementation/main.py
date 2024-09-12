import tensorflow as tf
from model import LSLM
from dataset import LSLMDataset, create_tf_dataset
from train import train, evaluate

VOCAB_SIZE = 10000
D_MODEL = 512
NHEAD = 8
NUM_LAYERS = 6
NUM_AUDIO_TOKENS = 1024
NUM_SAMPLES = 10000
NUM_EPOCHS = 10
BATCH_SIZE = 32

def main():
    # Check for GPU
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Create dataset and split into train, validation, and test
    full_dataset = LSLMDataset(num_samples=NUM_SAMPLES)
    train_size = int(0.8 * NUM_SAMPLES)
    val_size = int(0.1 * NUM_SAMPLES)

    train_dataset = create_tf_dataset(
        dataset=full_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_dataset = create_tf_dataset(
        dataset=full_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Initialize model
    model = LSLM(VOCAB_SIZE, D_MODEL, NHEAD, NUM_LAYERS, NUM_AUDIO_TOKENS)

    # Initialize optimizer and loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Training
    train(model, train_dataset, optimizer, loss_fn, NUM_EPOCHS)

    # Evaluation
    val_loss = evaluate(model, val_dataset, loss_fn)
    print(f"Validation Loss: {val_loss:.4f}")

    # Save model
    model.save_weights('lslm_model')

    print("Training and evaluation complete. Model saved.")

if __name__ == "__main__":
    main()
