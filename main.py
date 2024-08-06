import torch
from torch.utils.data import DataLoader, random_split
import torchaudio
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset and split into train, validation, and test
    full_dataset = LSLMDataset(num_samples=10000)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    model = LSLM(VOCAB_SIZE, D_MODEL, NHEAD, NUM_LAYERS, NUM_AUDIO_TOKENS).to(device)
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Training
    num_epochs = 10
    train(model, train_dataloader, optimizer, criterion, device, num_epochs)

    # Evaluation
    val_loss = evaluate(model, val_dataloader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}")

    # Test
    test_loss = evaluate(model, test_dataloader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")

    # Ablation studies
    fusion_types = ['early', 'middle', 'late']
    for fusion_type in fusion_types:
        model.fusion_module.fusion_type = fusion_type
        test_loss = evaluate(model, test_dataloader, criterion, device)
        print(f"Fusion type: {fusion_type}, Test Loss: {test_loss:.4f}")

    # Visualization
    sample_input = next(iter(test_dataloader))
    visualize_attention(model, sample_input)

    # Command-based FDM test
    command_test_data = [(torch.randn(MAX_AUDIO_LENGTH * SAMPLE_RATE), "Honey") for _ in range(100)]
    command = model.speaking_encoder.tokenizer("Honey", return_tensors="pt").input_values.to(device)
    command_based_fdm_test(model, command_test_data, command, device)

    # Voice-based FDM test
    voice_test_data = [(torch.randn(MAX_AUDIO_LENGTH * SAMPLE_RATE), torch.randn(SAMPLE_RATE)) for _ in range(100)]
    voice_based_fdm_test(model, voice_test_data, device)

    # Turn-taking analysis
    turn_taking_test_data = [(torch.randn(MAX_AUDIO_LENGTH * SAMPLE_RATE), torch.randint(0, 2, (1,)).item()) for _ in range(100)]
    analyze_turn_taking(model, turn_taking_test_data, device)

    # Generate speech from text
    test_text = "Hello, this is a test of the LSLM model."
    generated_speech = generate_speech(model, test_text, device)
    torchaudio.save('generated_speech.wav', torch.from_numpy(generated_speech), SAMPLE_RATE)

    # Visualize audio quantization
    test_audio = next(iter(test_dataloader))[0][0].to(device)
    visualize_quantization(model, test_audio)

    # Save model
    torch.save(model.state_dict(), 'lslm_model.pth')

    print("Training, evaluation, and testing complete. Model saved.")

if __name__ == "__main__":
    main()
