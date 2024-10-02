import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 10  # seconds

def add_noise(audio, snr_db):
    signal_power = audio.norm(p=2)
    noise = torch.randn_like(audio)
    noise_power = noise.norm(p=2)
    snr = 10**(snr_db/10)
    scale = snr * noise_power / signal_power
    noisy_audio = (scale * audio + noise) / (scale + 1)
    return noisy_audio

class LSLMDataset(Dataset):
    def __init__(self, num_samples=1000, max_length=MAX_AUDIO_LENGTH * SAMPLE_RATE):
        self.num_samples = num_samples
        self.max_length = max_length
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Generate dummy data
        self.tts_data = [(torch.randn(max_length), f"Sample text {i}") for i in range(num_samples)]
        self.interruption_data = [torch.randn(SAMPLE_RATE) for _ in range(100)]  # 100 different interruptions
        self.noise_data = [torch.randn(max_length) for _ in range(10)]  # 10 different noise patterns

    def __getitem__(self, idx):
        tts_sample, text = self.tts_data[idx]
        interruption = self.interruption_data[torch.randint(0, len(self.interruption_data), (1,))]
        noise = self.noise_data[torch.randint(0, len(self.noise_data), (1,))]
        
        # Combine TTS sample with interruption and noise
        combined_audio = tts_sample + F.pad(interruption, (0, self.max_length - interruption.shape[0])) + add_noise(tts_sample, snr_db=10)
        
        # Process audio
        audio_features = self.audio_processor(combined_audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.squeeze(0)
        
        # Tokenize text
        tokenized_text = self.text_tokenizer(text, return_tensors="pt", padding="max_length", max_length=50, truncation=True).input_ids.squeeze(0)
        
        return audio_features, tokenized_text

    def __len__(self):
        return self.num_samples

def collate_fn(batch):
    audio_features, tokenized_texts = zip(*batch)
    
    # Pad audio features
    audio_features = pad_sequence([torch.Tensor(sample) for sample in audio_features], batch_first=True)
    
    # Pad tokenized texts
    tokenized_texts = pad_sequence([torch.Tensor(text) for text in tokenized_texts], batch_first=True, padding_value=0)
    
    return audio_features, tokenized_texts