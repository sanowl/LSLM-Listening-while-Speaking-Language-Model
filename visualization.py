import torch
import matplotlib.pyplot as plt

def visualize_attention(model, sample_input):
    attention_weights = model.decoder.transformer_decoder.layers[-1].self_attn.attention_weights
    plt.figure(figsize=(10, 10))
    plt.imshow(attention_weights.cpu().numpy(), cmap='viridis')
    plt.title('Attention Weights Visualization')
    plt.xlabel('Query')
    plt.ylabel('Key')
    plt.colorbar()
    plt.show()

def visualize_quantization(model, audio_sample):
    model.eval()
    with torch.no_grad():
        quantized = model.speaking_encoder.audio_quantizer.quantize(audio_sample)
        plt.figure(figsize=(12, 6))
        plt.imshow(quantized.cpu().numpy(), aspect='auto', interpolation='nearest')
        plt.title('Audio Quantization Visualization')
        plt.xlabel('Time')
        plt.ylabel('Quantization Index')
        plt.colorbar()
        plt.show()
