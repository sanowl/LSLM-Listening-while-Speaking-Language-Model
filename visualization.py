import torch
import matplotlib.pyplot as plt

def visualize_attention(model, sample_input):
    model.eval()
    with torch.no_grad():
        audio_features, tokenized_text = sample_input
        device = next(model.parameters()).device
        audio_features = audio_features.to(device)
        tokenized_text = tokenized_text.to(device)

        speaking_feats = model.speaking_encoder(tokenized_text)
        listening_feats = model.listening_encoder(audio_features)
        fused = model.fusion_module(speaking_feats, listening_feats)

        layer = model.decoder.transformer_decoder.layers[-1]
        attn_output, attn_weights = layer.self_attn(
            fused.transpose(0, 1), fused.transpose(0, 1), fused.transpose(0, 1),
            need_weights=True
        )
        attn = attn_weights.mean(dim=0).cpu().numpy()

        plt.figure(figsize=(10, 10))
        plt.imshow(attn, cmap='viridis')
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
