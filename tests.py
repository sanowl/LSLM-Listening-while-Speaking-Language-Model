import torch

def command_based_fdm_test(model, test_data, command, device):
    model.eval()
    interruptions = 0
    total = len(test_data)
    with torch.no_grad():
        for sample, _ in test_data:
            sample = sample.unsqueeze(0).to(device)
            command = command.unsqueeze(0).to(device)
            output = model(sample, command, is_training=False)
            if output == model.irq_token:
                interruptions += 1
    
    print(f"Interruption rate: {interruptions/total:.2f}")

def voice_based_fdm_test(model, test_data, device):
    model.eval()
    interruptions = 0
    total = len(test_data)
    with torch.no_grad():
        for sample, voice_command in test_data:
            sample = sample.unsqueeze(0).to(device)
            voice_command = voice_command.unsqueeze(0).to(device)
            output = model(sample, voice_command, is_training=False)
            if output == model.irq_token:
                interruptions += 1
    
    print(f"Interruption rate: {interruptions/total:.2f}")

def analyze_turn_taking(model, test_data, device):
    model.eval()
    correct_detections = 0
    false_positives = 0
    false_negatives = 0
    
    with torch.no_grad():
        for sample, label in test_data:
            sample = sample.unsqueeze(0).to(device)
            output = model(sample, torch.zeros(1, 1, model.listening_encoder.d_model).to(device), is_training=False)
            
            if output == model.irq_token and label == 1:
                correct_detections += 1
            elif output == model.irq_token and label == 0:
                false_positives += 1
            elif output != model.irq_token and label == 1:
                false_negatives += 1
    
    precision = correct_detections / (correct_detections + false_positives)
    recall = correct_detections / (correct_detections + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    print(f"Turn-taking Performance:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

def generate_speech(model, text, device):
    model.eval()
    with torch.no_grad():
        tokenized_text = model.speaking_encoder.tokenizer(text, return_tensors="pt").input_values.to(device)
        generated_audio = model.generate(tokenized_text)
        return generated_audio.cpu().numpy()
