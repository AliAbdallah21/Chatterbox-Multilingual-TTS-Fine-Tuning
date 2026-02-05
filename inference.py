import os
import torch
import soundfile as sf
from peft import PeftModel
from safetensors.torch import load_file

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from config import FinetuneConfig


def load_finetuned_model(checkpoint_path, config):
    """Load model with finetuned LoRA weights - FIXED"""
    
    print(f"Loading base model...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=config.device)
    
    print(f"Setting up LoRA...")
    from peft import LoraConfig, get_peft_model
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=list(config.lora_target_modules),
    )
    
    # ✅ Apply LoRA (creates the adapter layers)
    model.t3 = get_peft_model(model.t3, lora_config)
    
    # ✅ Load the trained LoRA weights
    model_path = os.path.join(checkpoint_path, "model.safetensors")
    print(f"Loading checkpoint: {model_path}")
    
    lora_weights = load_file(model_path, device=config.device)
    
    # Load weights into the PEFT model
    model.t3.load_state_dict(lora_weights, strict=False)
    
    print(f"✅ Loaded {len(lora_weights)} LoRA parameters")
    
    # Verify
    lora_params = [p for n, p in model.t3.named_parameters() if 'lora' in n.lower()]
    print(f"✅ Model has {len(lora_params)} active LoRA parameters")
    
    model.t3.eval()
    model.s3gen.eval()
    model.ve.eval()
    
    return model


def generate(model, text, output_path, reference_audio=None, language_id="ar"):
    """Generate speech"""
    print(f"Generating: {text}")
    
    with torch.no_grad():
        if reference_audio:
            wav = model.generate(
                text,
                language_id=language_id,
                audio_prompt_path=reference_audio,
                exaggeration=0.5,
                cfg_weight=0.5,
                temperature=0.8,
            )
        else:
            wav = model.generate(
                text, 
                language_id=language_id,
                exaggeration=0.5,
                cfg_weight=0.5,
                temperature=0.8,
            )
    
    wav_np = wav.squeeze().cpu().numpy()
    sf.write(output_path, wav_np, model.sr)
    print(f"✅ Saved: {output_path} ({len(wav_np)/model.sr:.2f}s)")
    return len(wav_np) / model.sr


def test_checkpoint(checkpoint_name, reference_audio=None):
    """Test a specific checkpoint"""
    config = FinetuneConfig()
    
    checkpoint_path = os.path.join(config.output_dir, checkpoint_name)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"Available checkpoints:")
        if os.path.exists(config.output_dir):
            for item in os.listdir(config.output_dir):
                print(f"  - {item}")
        return
    
    # Load model
    model = load_finetuned_model(checkpoint_path, config)
    
    # Test sentences - Egyptian Arabic
    test_texts = [
        "الكتاب ده عبارة عن حكايات على القهوة",
        "مرحبا كيف حالك اليوم",
        "انا بحب مصر جدا",
        "ازيك يا باشا عامل ايه",
        "الجو حلو النهارده",
    ]
    
    # Create output directory
    output_dir = f"./test_outputs/{checkpoint_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"Testing checkpoint: {checkpoint_name}")
    print(f"Reference audio: {reference_audio or 'None (using built-in)'}")
    print(f"{'='*50}\n")
    
    total_duration = 0
    for i, text in enumerate(test_texts):
        output_path = os.path.join(output_dir, f"sample_{i+1}.wav")
        try:
            duration = generate(model, text, output_path, reference_audio)
            total_duration += duration
        except Exception as e:
            print(f"❌ Error generating sample {i+1}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print(f"✅ All samples saved in: {output_dir}")
    print(f"Total audio duration: {total_duration:.2f}s")
    print(f"{'='*50}")


def test_base_model(reference_audio=None):
    """Test the base model without finetuning for comparison"""
    config = FinetuneConfig()
    
    print("Loading base model (no finetuning)...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=config.device)
    
    test_texts = [
        "الكتاب ده عبارة عن حكايات على القهوة",
        "مرحبا كيف حالك اليوم",
        "انا بحب مصر جدا",
    ]
    
    output_dir = "./test_outputs/base_model"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"Testing BASE model (no finetuning)")
    print(f"{'='*50}\n")
    
    for i, text in enumerate(test_texts):
        output_path = os.path.join(output_dir, f"sample_{i+1}.wav")
        try:
            generate(model, text, output_path, reference_audio)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n✅ Base model samples saved in: {output_dir}")


def compare_checkpoints(checkpoints, reference_audio=None):
    """Compare multiple checkpoints side by side"""
    config = FinetuneConfig()
    
    test_text = "الكتاب ده عبارة عن حكايات على القهوة"
    
    output_dir = "./test_outputs/comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"Comparing checkpoints with text:")
    print(f"  \"{test_text}\"")
    print(f"{'='*50}\n")
    
    # First, test base model
    print("1. Testing base model...")
    try:
        model = ChatterboxMultilingualTTS.from_pretrained(device=config.device)
        output_path = os.path.join(output_dir, "base_model.wav")
        generate(model, test_text, output_path, reference_audio)
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Base model error: {e}")
    
    # Then test each checkpoint
    for i, ckpt in enumerate(checkpoints):
        print(f"\n{i+2}. Testing {ckpt}...")
        checkpoint_path = os.path.join(config.output_dir, ckpt)
        
        if not os.path.exists(checkpoint_path):
            print(f"❌ Not found: {checkpoint_path}")
            continue
        
        try:
            model = load_finetuned_model(checkpoint_path, config)
            output_path = os.path.join(output_dir, f"{ckpt}.wav")
            generate(model, test_text, output_path, reference_audio)
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print(f"✅ Comparison samples saved in: {output_dir}")
    print(f"{'='*50}")


def list_checkpoints():
    """List all available checkpoints"""
    config = FinetuneConfig()
    
    print(f"\n{'='*50}")
    print(f"Available checkpoints in: {config.output_dir}")
    print(f"{'='*50}")
    
    if not os.path.exists(config.output_dir):
        print("❌ Output directory doesn't exist yet")
        return []
    
    checkpoints = []
    for item in sorted(os.listdir(config.output_dir)):
        item_path = os.path.join(config.output_dir, item)
        if os.path.isdir(item_path):
            model_file = os.path.join(item_path, "model.safetensors")
            if os.path.exists(model_file):
                size_mb = os.path.getsize(model_file) / (1024 * 1024)
                print(f"  ✅ {item} ({size_mb:.1f} MB)")
                checkpoints.append(item)
            else:
                print(f"  ⚠️ {item} (no model.safetensors)")
    
    if not checkpoints:
        print("  No checkpoints found yet")
    
    return checkpoints


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test finetuned Chatterbox model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint name (e.g., checkpoint-500)")
    parser.add_argument("--reference", type=str, default=None, help="Reference audio path for voice cloning")
    parser.add_argument("--list", action="store_true", help="List available checkpoints")
    parser.add_argument("--base", action="store_true", help="Test base model without finetuning")
    parser.add_argument("--compare", action="store_true", help="Compare multiple checkpoints")
    parser.add_argument("--text", type=str, default=None, help="Custom text to generate")
    
    args = parser.parse_args()
    
    # Default reference audio
    default_ref = "C:/Users/11/chatterbox-finetuning/MyTTSDataset_Dahih/wavs/dahih_00006.wav"
    reference = args.reference or default_ref
    
    if not os.path.exists(reference):
        print(f"⚠️ Reference audio not found: {reference}")
        reference = None
    
    if args.list:
        list_checkpoints()
    
    elif args.base:
        test_base_model(reference)
    
    elif args.compare:
        checkpoints = list_checkpoints()
        if checkpoints:
            compare_checkpoints(checkpoints, reference)
    
    elif args.checkpoint:
        test_checkpoint(args.checkpoint, reference)
    
    else:
        # Default: list checkpoints and test latest
        checkpoints = list_checkpoints()
        if checkpoints:
            latest = checkpoints[-1]
            print(f"\nTesting latest checkpoint: {latest}")
            test_checkpoint(latest, reference)
        else:
            print("\nNo checkpoints available. Testing base model instead...")
            test_base_model(reference)