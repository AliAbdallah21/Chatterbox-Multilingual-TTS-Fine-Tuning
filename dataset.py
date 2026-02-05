import os
import torch
import pandas as pd
import soundfile as sf
import torchaudio
from torch.utils.data import Dataset

class EgyptianArabicDataset(Dataset):
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.sr = model.sr
        
        # Load metadata
        metadata_path = os.path.join(config.data_dir, "metadata.csv")
        self.df = pd.read_csv(metadata_path, sep="|", header=0)
        # NEW (handles 2 columns):
        if len(self.df.columns) == 2:
            self.df.columns = ["filename", "raw_text"]
            self.df["normalized_text"] = self.df["raw_text"]  # Same as raw
        elif len(self.df.columns) == 3:
            self.df.columns = ["filename", "raw_text", "normalized_text"]
        else:
            raise ValueError(f"Expected 2 or 3 columns, got {len(self.df.columns)}")
        
        # Wav directory
        self.wav_dir = os.path.join(config.data_dir, "wavs")
        
        # Filter valid files
        valid_rows = []
        for idx, row in self.df.iterrows():
            wav_path = os.path.join(self.wav_dir, row["filename"])
            if os.path.exists(wav_path):
                valid_rows.append(row)
        
        self.df = pd.DataFrame(valid_rows)
        print(f"Dataset loaded: {len(self.df)} samples")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load audio
        wav_path = os.path.join(self.wav_dir, row["filename"])
        wav, sr = sf.read(wav_path, dtype='float32')
        
        # Convert to tensor
        wav = torch.from_numpy(wav)
        
        # Handle stereo -> mono
        if wav.dim() > 1:
            wav = wav.mean(dim=-1)
        
        # Resample if needed (use torchaudio for quality)
        if sr != self.sr:
            wav = wav.unsqueeze(0)  # Add channel dim
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            wav = resampler(wav).squeeze(0)
        
        # NO truncation needed - chunks are already 2-4s!
        
        # Get text
        text = row["normalized_text"]
        
        return {
            "audio": wav,
            "text": text,
            "filename": row["filename"]
        }


def collate_fn(batch):
    """Collate batch with padding"""
    # Find max audio length
    max_len = max(item["audio"].shape[0] for item in batch)
    
    # Pad audio
    audios = []
    for item in batch:
        audio = item["audio"]
        if audio.shape[0] < max_len:
            padding = torch.zeros(max_len - audio.shape[0])
            audio = torch.cat([audio, padding])
        audios.append(audio)
    
    return {
        "audio": torch.stack(audios),
        "text": [item["text"] for item in batch],
        "filename": [item["filename"] for item in batch]
    }