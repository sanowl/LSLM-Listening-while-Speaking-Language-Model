import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax.training.common_utils import shard
from tqdm import tqdm
from functools import partial
import numpy as np
from model import LSLM
from dataset import LSLMDataset, collate_fn
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

def create_train_state(rng, model, learning_rate):
    """Creates initial training state."""
    params = model.init(rng, jnp.ones((1, MAX_AUDIO_LENGTH * SAMPLE_RATE)))
    tx = optax.adamw(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def split_dataset(dataset, train_size, val_size):
    """Split the dataset into training, validation, and test sets."""
    indices = jax.random.permutation(jax.random.PRNGKey(0), len(dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    test_dataset = [dataset[i] for i in test_indices]
    
    return train_dataset, val_dataset, test_dataset

def data_loader(dataset, batch_size, shuffle=False):
    """Simple data loader function."""
    data_size = len(dataset)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, data_size, batch_size):
        batch_indices = indices[start_idx:start_idx+batch_size]
        batch = [dataset[i] for i in batch_indices]
        yield collate_fn(batch)

def main():
    rng = jax.random.PRNGKey(0)
    
    # Create dataset and split into train, validation, and test
    full_dataset = LSLMDataset(num_samples=10000)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset, train_size, val_size)

    # Create dataloaders
    train_dataloader = data_loader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = data_loader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = data_loader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    model = LSLM(VOCAB_SIZE, D_MODEL, NHEAD, NUM_LAYERS, NUM_AUDIO_TOKENS)
    state = create_train_state(rng, model, learning_rate=5e-4)
    
    # Training
    num_epochs = 10
    state = train(state, train_dataloader, rng, num_epochs)

    # Evaluation
    val_loss = evaluate(state, val_dataloader)
    print(f"Validation Loss: {val_loss:.4f}")

    # Test
    test_loss = evaluate(state, test_dataloader)
    print(f"Test Loss: {test_loss:.4f}")

    # Ablation studies
    fusion_types = ['early', 'middle', 'late']
    for fusion_type in fusion_types:
        model.fusion_module.fusion_type = fusion_type
        test_loss = evaluate(state, test_dataloader)
        print(f"Fusion type: {fusion_type}, Test Loss: {test_loss:.4f}")

    # Visualization
    sample_input = next(iter(test_dataloader))
    visualize_attention(model, state.params, sample_input)

    # Command-based FDM test
    command_test_data = [(jax.random.normal(rng, (MAX_AUDIO_LENGTH * SAMPLE_RATE,)), "Honey") for _ in range(100)]
    command = model.speaking_encoder.tokenizer("Honey", return_tensors="jax").input_values
    command_based_fdm_test(model, state.params, command_test_data, command)

    # Voice-based FDM test
    voice_test_data = [(jax.random.normal(rng, (MAX_AUDIO_LENGTH * SAMPLE_RATE,)), jax.random.normal(rng, (SAMPLE_RATE,))) for _ in range(100)]
    voice_based_fdm_test(model, state.params, voice_test_data)

    # Turn-taking analysis
    turn_taking_test_data = [(jax.random.normal(rng, (MAX_AUDIO_LENGTH * SAMPLE_RATE,)), jax.random.randint(rng, (1,), 0, 2).item()) for _ in range(100)]
    analyze_turn_taking(model, state.params, turn_taking_test_data)

    # Generate speech from text
    test_text = "Hello, this is a test of the LSLM model."
    generated_speech = generate_speech(model, state.params, test_text)
    torchaudio.save('generated_speech.wav', jax.device_get(generated_speech), SAMPLE_RATE)

    # Visualize audio quantization
    test_audio = next(iter(test_dataloader))[0][0]
    visualize_quantization(model, state.params, test_audio)

    # Save model
    jax.numpy.savez('lslm_model.npz', *jax.tree_leaves(state.params))

    print("Training, evaluation, and testing complete. Model saved.")

if __name__ == "__main__":
    main()
