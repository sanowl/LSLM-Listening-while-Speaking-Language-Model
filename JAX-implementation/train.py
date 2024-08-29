import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

def train(state, dataloader, rng, num_epochs):
    """Training loop for the model using JAX and Flax."""
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            tts_samples, combined_audios, tokenized_texts = batch

            def loss_fn(params):
                outputs = state.apply_fn({'params': params}, tokenized_texts, combined_audios)
                loss = optax.softmax_cross_entropy(outputs.view(-1, outputs.shape[-1]), tts_samples.view(-1))
                return loss.mean()

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    return state

def evaluate(state, dataloader):
    """Evaluation loop for the model using JAX and Flax."""
    total_loss = 0
    for batch in dataloader:
        tts_samples, combined_audios, tokenized_texts = batch

        def loss_fn(params):
            outputs = state.apply_fn({'params': params}, tokenized_texts, combined_audios, is_training=False)
            loss = optax.softmax_cross_entropy(outputs.view(-1, outputs.shape[-1]), tts_samples.view(-1))
            return loss.mean()

        loss = loss_fn(state.params)
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
