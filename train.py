import torch
from tqdm import tqdm

def train(model, dataloader, optimizer, criterion, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            tts_samples, combined_audios, tokenized_texts = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            output = model(tokenized_texts, combined_audios)
            loss = criterion(output.view(-1, output.size(-1)), tts_samples.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            tts_samples, combined_audios, tokenized_texts = [b.to(device) for b in batch]
            
            output = model(tokenized_texts, combined_audios, is_training=False)
            loss = criterion(output.view(-1, output.size(-1)), tts_samples.view(-1))
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

