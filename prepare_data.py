"""
Prepare Dahih Arabic TTS Dataset - CLEANED VERSION
Filters English segments, validates audio, creates proper metadata
For 114 hours of pure Arabic data
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import librosa
import numpy as np
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

SOURCE_DIR = r"path to the source directory containing playlists"
OUTPUT_DIR = r"path to the output cleaned dataset directory"

# Audio quality thresholds
MIN_DURATION = 1.0  # seconds
MAX_DURATION = 15.0  # seconds
TARGET_SR = 24000  # Chatterbox sample rate

# Text validation
MIN_TEXT_LENGTH = 3  # minimum characters
MAX_TEXT_LENGTH = 500  # maximum characters

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def has_english(text):
    """Check if text contains English letters"""
    return bool(re.search(r'[a-zA-Z]', text))

def is_valid_audio(audio_path, min_dur=MIN_DURATION, max_dur=MAX_DURATION):
    """
    Validate audio file:
    - Exists and is readable
    - Duration within range
    - Not silent/corrupted
    """
    try:
        info = sf.info(audio_path)
        duration = info.duration
        
        if duration < min_dur or duration > max_dur:
            return False, f"Duration {duration:.1f}s out of range"
        
        # Check if audio is silent
        audio, sr = sf.read(audio_path)
        if np.max(np.abs(audio)) < 0.001:
            return False, "Audio is silent"
        
        return True, "OK"
        
    except Exception as e:
        return False, f"Read error: {str(e)}"

def is_valid_text(text, min_len=MIN_TEXT_LENGTH, max_len=MAX_TEXT_LENGTH):
    """
    Validate text:
    - Not empty
    - Length within range
    - No English characters
    """
    text = text.strip()
    
    if len(text) < min_len:
        return False, f"Too short ({len(text)} chars)"
    
    if len(text) > max_len:
        return False, f"Too long ({len(text)} chars)"
    
    if has_english(text):
        return False, "Contains English"
    
    # Check if mostly Arabic
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    if arabic_chars < len(text) * 0.3:  # At least 30% Arabic
        return False, "Not enough Arabic characters"
    
    return True, "OK"

# ============================================================================
# MAIN PREPARATION
# ============================================================================

def prepare_dataset():
    """
    Scan all segments, filter clean Arabic, prepare TTS dataset
    """
    
    print("="*70)
    print("🎯 PREPARING CLEAN ARABIC TTS DATASET")
    print("="*70)
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    # Create output directories
    output_wavs = os.path.join(OUTPUT_DIR, "wavs")
    os.makedirs(output_wavs, exist_ok=True)
    
    # Statistics
    stats = {
        "total_segments": 0,
        "valid_segments": 0,
        "rejected_english": 0,
        "rejected_audio": 0,
        "rejected_text": 0,
        "total_duration": 0.0,
    }
    
    metadata = []
    file_counter = 0
    
    print("🔍 Scanning all playlists and videos...")
    print()
    
    # Walk through all playlists
    for playlist_dir in os.listdir(SOURCE_DIR):
        playlist_path = os.path.join(SOURCE_DIR, playlist_dir)
        
        if not os.path.isdir(playlist_path):
            continue
        
        print(f"📂 {playlist_dir}")
        
        # Walk through all video folders
        for video_dir_name in os.listdir(playlist_path):
            video_dir = os.path.join(playlist_path, video_dir_name)
            
            if not os.path.isdir(video_dir):
                continue
            
            # Process segments in this video
            segment_dirs = sorted([d for d in os.listdir(video_dir) 
                                 if d.startswith("segment_") and 
                                 os.path.isdir(os.path.join(video_dir, d))])
            
            for segment_name in segment_dirs:
                stats["total_segments"] += 1
                segment_dir = os.path.join(video_dir, segment_name)
                
                audio_path = os.path.join(segment_dir, "audio.wav")
                transcript_path = os.path.join(segment_dir, "transcript.txt")
                
                # Check files exist
                if not os.path.exists(audio_path) or not os.path.exists(transcript_path):
                    continue
                
                # Read transcript
                with open(transcript_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                
                # Validate text
                text_valid, text_reason = is_valid_text(text)
                if not text_valid:
                    if "English" in text_reason:
                        stats["rejected_english"] += 1
                    else:
                        stats["rejected_text"] += 1
                    continue
                
                # Validate audio
                audio_valid, audio_reason = is_valid_audio(audio_path)
                if not audio_valid:
                    stats["rejected_audio"] += 1
                    continue
                
                # ✅ Valid sample - add to dataset
                file_counter += 1
                new_filename = f"dahih_{file_counter:06d}.wav"
                new_audio_path = os.path.join(output_wavs, new_filename)
                
                # Copy and optionally resample audio
                audio, sr = sf.read(audio_path)
                
                # Resample if needed
                if sr != TARGET_SR:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
                
                # Normalize audio (prevent clipping)
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    audio = audio / max_val * 0.95
                
                # Save
                sf.write(new_audio_path, audio, TARGET_SR)
                
                # Add to metadata
                duration = len(audio) / TARGET_SR
                stats["total_duration"] += duration
                
                metadata.append({
                    "filename": new_filename,
                    "text": text,
                    "duration": duration,
                    "source_video": video_dir_name,
                    "source_segment": segment_name
                })
                
                stats["valid_segments"] += 1
                
                # Progress update
                if file_counter % 1000 == 0:
                    print(f"  ✅ Processed {file_counter:,} valid segments...")
    
    # Save metadata
    print()
    print("💾 Saving metadata...")
    
    df = pd.DataFrame(metadata)
    
    # Save full metadata (for reference)
    df.to_csv(os.path.join(OUTPUT_DIR, "metadata_full.csv"), index=False, encoding="utf-8")
    
    # Save LJSpeech-style metadata (for training)
    # Format: filename|text
    with open(os.path.join(OUTPUT_DIR, "metadata.csv"), "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(f"{row['filename']}|{row['text']}\n")
    
    # Print summary
    print()
    print("="*70)
    print("📊 DATASET PREPARATION SUMMARY")
    print("="*70)
    print(f"Total segments scanned: {stats['total_segments']:,}")
    print(f"✅ Valid segments: {stats['valid_segments']:,} ({stats['valid_segments']/stats['total_segments']*100:.1f}%)")
    print()
    print(f"❌ Rejected:")
    print(f"  - Contains English: {stats['rejected_english']:,}")
    print(f"  - Invalid audio: {stats['rejected_audio']:,}")
    print(f"  - Invalid text: {stats['rejected_text']:,}")
    print()
    print(f"⏱️  Total duration: {stats['total_duration']/3600:.2f} hours")
    print(f"📁 Average segment: {stats['total_duration']/stats['valid_segments']:.1f} seconds")
    print()
    print(f"💾 Output directory: {OUTPUT_DIR}")
    print(f"  - wavs/: {stats['valid_segments']:,} audio files")
    print(f"  - metadata.csv: Training metadata")
    print(f"  - metadata_full.csv: Full metadata with sources")
    print("="*70)
    
    return df, stats

# ============================================================================
# VALIDATION REPORT
# ============================================================================

def generate_validation_report(df, stats, output_dir):
    """Generate detailed validation report"""
    
    report_path = os.path.join(output_dir, "dataset_report.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write("CLEAN ARABIC TTS DATASET - VALIDATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        
        f.write("STATISTICS:\n")
        f.write(f"  Total samples: {len(df):,}\n")
        f.write(f"  Total duration: {stats['total_duration']/3600:.2f} hours\n")
        f.write(f"  Average duration: {df['duration'].mean():.2f}s\n")
        f.write(f"  Min duration: {df['duration'].min():.2f}s\n")
        f.write(f"  Max duration: {df['duration'].max():.2f}s\n\n")
        
        f.write("TEXT STATISTICS:\n")
        f.write(f"  Average text length: {df['text'].str.len().mean():.1f} chars\n")
        f.write(f"  Min text length: {df['text'].str.len().min()} chars\n")
        f.write(f"  Max text length: {df['text'].str.len().max()} chars\n\n")
        
        f.write("DURATION DISTRIBUTION:\n")
        bins = [0, 3, 6, 9, 12, 15]
        for i in range(len(bins)-1):
            count = len(df[(df['duration'] >= bins[i]) & (df['duration'] < bins[i+1])])
            f.write(f"  {bins[i]}-{bins[i+1]}s: {count:,} ({count/len(df)*100:.1f}%)\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"📄 Validation report saved: {report_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n🚀 Starting dataset preparation...\n")
    
    # Prepare dataset
    df, stats = prepare_dataset()
    
    # Generate validation report
    generate_validation_report(df, stats, OUTPUT_DIR)
    
    print("\n✅ Dataset preparation complete!")
    print(f"\n💡 Next steps:")
    print(f"   1. Review: {OUTPUT_DIR}/dataset_report.txt")
    print(f"   2. Update config.py: data_dir = '{OUTPUT_DIR}'")
    print(f"   3. Run training: python train.py")
    print()