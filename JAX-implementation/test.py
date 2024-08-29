import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state

def command_based_fdm_test(model, test_data, command, params):
    interruptions = 0
    total = len(test_data)
    for sample, _ in test_data:
        sample = jnp.expand_dims(sample, axis=0)
        command = jnp.expand_dims(command, axis=0)
        output = model.apply(params, sample, command, is_training=False)
        if jnp.argmax(output, axis=-1) == model.irq_token:
            interruptions += 1
    
    print(f"Interruption rate: {interruptions / total:.2f}")

def voice_based_fdm_test(model, test_data, params):
    interruptions = 0
    total = len(test_data)
    for sample, voice_command in test_data:
        sample = jnp.expand_dims(sample, axis=0)
        voice_command = jnp.expand_dims(voice_command, axis=0)
        output = model.apply(params, sample, voice_command, is_training=False)
        if jnp.argmax(output, axis=-1) == model.irq_token:
            interruptions += 1
    
    print(f"Interruption rate: {interruptions / total:.2f}")

def analyze_turn_taking(model, test_data, params):
    correct_detections = 0
    false_positives = 0
    false_negatives = 0
    
    for sample, label in test_data:
        sample = jnp.expand_dims(sample, axis=0)
        zero_command = jnp.zeros((1, 1, model.listening_encoder.d_model))
        output = model.apply(params, sample, zero_command, is_training=False)
        
        if jnp.argmax(output, axis=-1) == model.irq_token and label == 1:
            correct_detections += 1
        elif jnp.argmax(output, axis=-1) == model.irq_token and label == 0:
            false_positives += 1
        elif jnp.argmax(output, axis=-1) != model.irq_token and label == 1:
            false_negatives += 1
    
    precision = correct_detections / (correct_detections + false_positives)
    recall = correct_detections / (correct_detections + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    print(f"Turn-taking Performance:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

def generate_speech(model, text, params):
    tokenized_text = model.speaking_encoder.tokenizer(text, return_tensors="jax").input_values
    generated_audio = model.apply(params, tokenized_text, is_training=False, method=model.generate)
    return np.array(jax.device_get(generated_audio))
