import tensorflow as tf
from tqdm import tqdm

def train(model, dataset, optimizer, loss_fn, num_epochs):
    @tf.function
    def train_step(tts_samples, combined_audios, tokenized_texts):
        with tf.GradientTape() as tape:
            output = model([tokenized_texts, combined_audios], training=True)
            loss = loss_fn(tts_samples, output)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(dataset, desc=f"Epoch {epoch+1}/{num_epochs}"):
            tts_samples, combined_audios, tokenized_texts = batch
            loss = train_step(tts_samples, combined_audios, tokenized_texts)
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

def evaluate(model, dataset, loss_fn):
    @tf.function
    def eval_step(tts_samples, combined_audios, tokenized_texts):
        output = model([tokenized_texts, combined_audios], training=False)
        return loss_fn(tts_samples, output)

    total_loss = 0
    num_batches = 0

    for batch in dataset:
        tts_samples, combined_audios, tokenized_texts = batch
        loss = eval_step(tts_samples, combined_audios, tokenized_texts)
        total_loss += loss
        num_batches += 1

    return total_loss / num_batches