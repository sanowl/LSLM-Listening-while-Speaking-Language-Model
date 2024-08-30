import tensorflow as tf

def command_based_fdm_test(model, test_data, command):
    interruptions = 0
    total = len(test_data)
    
    for sample, _ in test_data:
        sample = tf.expand_dims(sample, 0)
        command = tf.expand_dims(command, 0)
        output = model([sample, command], training=False)
        if tf.reduce_all(output == model.irq_token):
            interruptions += 1
    
    print(f"Interruption rate: {interruptions/total:.2f}")

def voice_based_fdm_test(model, test_data):
    interruptions = 0
    total = len(test_data)
    
    for sample, voice_command in test_data:
        sample = tf.expand_dims(sample, 0)
        voice_command = tf.expand_dims(voice_command, 0)
        output = model([sample, voice_command], training=False)
        if tf.reduce_all(output == model.irq_token):
            interruptions += 1
    
    print(f"Interruption rate: {interruptions/total:.2f}")

def analyze_turn_taking(model, test_data):
    correct_detections = 0
    false_positives = 0
    false_negatives = 0
    
    for sample, label in test_data:
        sample = tf.expand_dims(sample, 0)
        zero_input = tf.zeros((1, 1, model.listening_encoder.d_model))
        output = model([sample, zero_input], training=False)
        
        if tf.reduce_all(output == model.irq_token) and label == 1:
            correct_detections += 1
        elif tf.reduce_all(output == model.irq_token) and label == 0:
            false_positives += 1
        elif not tf.reduce_all(output == model.irq_token) and label == 1:
            false_negatives += 1
    
    precision = correct_detections / (correct_detections + false_positives + 1e-8)
    recall = correct_detections / (correct_detections + false_negatives + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    print(f"Turn-taking Performance:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

def generate_speech(model, text):
    tokenized_text = model.speaking_encoder.tokenizer(text, return_tensors="tf").input_values
    generated_audio = model.generate(tokenized_text)
    return generated_audio.numpy()
