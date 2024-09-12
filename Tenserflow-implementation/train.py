import tensorflow as tf
from tqdm import tqdm

def train(model, dataset, optimizer, loss_fn, num_epochs):
    """
    Training loop for the model.

    Args:
        model: The LSLM model instance.
        dataset: The training dataset.
        optimizer: The optimizer instance.
        loss_fn: The loss function.
        num_epochs: Number of epochs to train.
    """
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    @tf.function
    def train_step(speaking_input, listening_input, target):
        with tf.GradientTape() as tape:
            predictions = model((speaking_input, listening_input), training=True)
            loss = loss_fn(target, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)

    for epoch in range(num_epochs):
        train_loss.reset_states()
        for batch in tqdm(dataset, desc=f"Epoch {epoch+1}/{num_epochs}"):
            tts_samples, combined_audios, tokenized_texts = batch
            train_step(tokenized_texts, combined_audios, tokenized_texts)
        print(f"Epoch {epoch+1}, Loss: {train_loss.result():.4f}")

def evaluate(model, dataset, loss_fn):
    """
    Evaluation loop for the model.

    Args:
        model: The LSLM model instance.
        dataset: The evaluation dataset.
        loss_fn: The loss function.
    """
    eval_loss = tf.keras.metrics.Mean(name='eval_loss')

    @tf.function
    def eval_step(speaking_input, listening_input, target):
        predictions = model((speaking_input, listening_input), training=False)
        loss = loss_fn(target, predictions)
        eval_loss(loss)

    for batch in dataset:
        tts_samples, combined_audios, tokenized_texts = batch
        eval_step(tokenized_texts, combined_audios, tokenized_texts)
    return eval_loss.result()
