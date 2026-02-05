"""
Quick test for partial fine-tune checkpoints
No LoRA - just loads weights directly!
"""
import os
import torch
import soundfile as sf
from safetensors.torch import load_file
from chatterbox.mtl_tts import ChatterboxMultilingualTTS


# =========================================================================
# CONFIGURATION
# =========================================================================

# Test sentences
TEST_TEXTS = [
    "الكتاب ده عبارة عن حكايات على القهوة",
    "انا بحب مصر جدا",
    "ازيك يا باشا عامل ايه",
    "الجو حلو النهارده",
    "انا عايز اعرف اذا كان في شقق تانية بتمن اقل من دي",
    "اللغة العربية لغة جميلة وغنية بالتاريخ و الثقافة والادب و الحضارة العريقة زي الجمال بتاعها",
    "سارة دي مزة المزز"

]

# Reference audio (الدحيح voice)
REFERENCE_AUDIO = r"C:\Users\11\Documents\audio.wav"

# Output directory
OUTPUT_DIR = "./output_data_egyptian_partial"


# =========================================================================
# MAIN
# =========================================================================

def test_checkpoint(checkpoint_name):
    """Test a checkpoint"""
    
    # Build path
    checkpoint_path = os.path.join(OUTPUT_DIR, checkpoint_name, "model.safetensors")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"🎤 Testing Checkpoint: {checkpoint_name}")
    print(f"{'='*70}\n")
    
    # Load base model
    print("Loading base model...")
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
    
    # Load checkpoint weights
    print(f"Loading checkpoint: {checkpoint_path}")
    weights = load_file(checkpoint_path, device="cuda")
    model.t3.load_state_dict(weights, strict=False)
    
    print(f"  Loaded {len(weights)} parameters\n")
    
    # Set to eval
    model.t3.eval()
    model.s3gen.eval()
    model.ve.eval()
    
    # Create output directory
    output_dir = os.path.join("./test_outputs", checkpoint_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 Output: {output_dir}\n")
    print(f"{'─'*70}")
    
    # Generate samples
    for idx, text in enumerate(TEST_TEXTS, start=1):
        print(f"({idx}/{len(TEST_TEXTS)}) {text}")
        
        try:
            with torch.no_grad():
                wav = model.generate(
                    text,
                    language_id="ar",
                    audio_prompt_path=REFERENCE_AUDIO,
                    exaggeration=0.5,
                    cfg_weight=0.5,
                    temperature=0.8,
                )
            
            # Save
            filename = f"sample_{idx:02d}.wav"
            output_path = os.path.join(output_dir, filename)
            
            wav_np = wav.squeeze().cpu().numpy()
            sf.write(output_path, wav_np, model.sr)
            
            duration = len(wav_np) / model.sr
            print(f"  ✅ {filename} ({duration:.2f}s)")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n{'='*70}")
    print(f"🎉 Testing Complete!")
    print(f"📂 Samples saved in: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test partial fine-tune checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="epoch_1",
        help="Checkpoint name (e.g., 'checkpoint-2000', 'epoch_0', 'final_model')"
    )
    
    args = parser.parse_args()
    test_checkpoint(args.checkpoint)