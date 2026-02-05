import os
import torch
import logging
import librosa
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from safetensors.torch import save_file, load_file
from tqdm import tqdm
import torch.nn.functional as F

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.s3tokenizer import S3_SR
from chatterbox.models.t3.modules.cond_enc import T3Cond

from config import FinetuneConfig
from dataset import EgyptianArabicDataset, collate_fn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def freeze_strategically(model, config):
    """
    Strategic freezing for Egyptian dialect learning
    
    FREEZE:
    - Text encoder (protect phoneme knowledge)
    - Early acoustic layers (protect low-level audio)
    - S3Gen codec (protect audio quality)
    - VoiceEncoder (protect speaker embedding)
    
    UNFREEZE:
    - Prosody/duration predictors (learn Egyptian rhythm!)
    - Late acoustic decoder layers (learn speaker + dialect)
    """
    
    # 1. ALWAYS freeze codec and voice encoder
    for param in model.s3gen.parameters():
        param.requires_grad = False
    for param in model.ve.parameters():
        param.requires_grad = False
    
    logger.info("✅ Frozen: S3Gen codec, VoiceEncoder")
    
    # 2. Freeze/unfreeze T3 layers strategically
    total_params = 0
    trainable_params = 0
    
    for name, param in model.t3.named_parameters():
        
        # FREEZE text encoder (protect phoneme knowledge)
        if config.freeze_text_encoder and ('text_enc' in name or 'text_emb' in name):
            param.requires_grad = False
            total_params += param.numel()
            continue
        
        # UNFREEZE prosody/duration (learn Egyptian rhythm!)
        if config.unfreeze_prosody and ('duration' in name or 'prosody' in name or 'pitch' in name):
            param.requires_grad = True
            total_params += param.numel()
            trainable_params += param.numel()
            continue
        
        # Handle decoder layers strategically
        if 'decoder' in name or 'dec_layer' in name:
            # Extract layer number if possible
            try:
                # Try to find layer number in name
                import re
                layer_match = re.search(r'layer[._](\d+)|\.(\d+)\.', name)
                if layer_match:
                    layer_num = int(layer_match.group(1) or layer_match.group(2))
                    
                    # Freeze early layers (0-11), unfreeze late layers (12+)
                    if config.unfreeze_late_acoustic and layer_num >= 12:
                        param.requires_grad = True
                        trainable_params += param.numel()
                    elif config.freeze_early_acoustic:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                        trainable_params += param.numel()
                else:
                    # If can't parse layer number, use config default
                    if config.unfreeze_late_acoustic:
                        param.requires_grad = True
                        trainable_params += param.numel()
                    else:
                        param.requires_grad = False
            except:
                # Fallback: unfreeze if unfreeze_late_acoustic is True
                if config.unfreeze_late_acoustic:
                    param.requires_grad = True
                    trainable_params += param.numel()
                else:
                    param.requires_grad = False
            
            total_params += param.numel()
            continue
        
        # Default: keep trainable
        param.requires_grad = True
        total_params += param.numel()
        trainable_params += param.numel()
    
    logger.info(f"📊 Partial Fine-Tune Strategy:")
    logger.info(f"  Total T3 params: {total_params:,}")
    logger.info(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    logger.info(f"  Text encoder: {'FROZEN ❄️' if config.freeze_text_encoder else 'Trainable'}")
    logger.info(f"  Early acoustic: {'FROZEN ❄️' if config.freeze_early_acoustic else 'Trainable'}")
    logger.info(f"  Prosody: {'Trainable 🔥' if config.unfreeze_prosody else 'FROZEN'}")
    logger.info(f"  Late acoustic: {'Trainable 🔥' if config.unfreeze_late_acoustic else 'FROZEN'}")


def compute_loss(model, prepared, config):
    """Compute loss - same as before"""
    IGNORE_ID = -100
    
    t3_cond = prepared["t3_cond"]
    text_tokens = prepared["text_tokens"]
    text_token_lens = prepared["text_token_lens"]
    speech_tokens = prepared["speech_tokens"]
    speech_token_lens = prepared["speech_token_lens"]
    
    max_speech_vocab = model.t3.hp.speech_tokens_dict_size
    
    if speech_tokens.max() >= max_speech_vocab:
        logger.error(f"🚨 Invalid speech token: {speech_tokens.max()} >= {max_speech_vocab}")
        device = speech_tokens.device
        return torch.tensor(0.0, device=device, requires_grad=True), \
               torch.tensor(0.0, device=device, requires_grad=True)
    
    try:
        out = model.t3.forward(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            training=True,
        )
    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        device = speech_tokens.device
        return torch.tensor(0.0, device=device, requires_grad=True), \
               torch.tensor(0.0, device=device, requires_grad=True)
    
    device = out.text_logits.device
    
    # Text loss
    logits_text = out.text_logits[:, :-1, :]
    target_text = text_tokens[:, 1:]
    
    mask_text = torch.arange(target_text.size(1), device=device)[None, :] < (text_token_lens - 1)[:, None]
    target_text = target_text.masked_fill(~mask_text, IGNORE_ID)
    
    loss_text = F.cross_entropy(
        logits_text.reshape(-1, logits_text.size(-1)),
        target_text.reshape(-1),
        ignore_index=IGNORE_ID
    )
    
    # Speech loss
    logits_speech = out.speech_logits[:, :-1, :]
    target_speech = speech_tokens[:, 1:]
    
    mask_speech = torch.arange(target_speech.size(1), device=device)[None, :] < (speech_token_lens - 1)[:, None]
    target_speech = target_speech.masked_fill(~mask_speech, IGNORE_ID)
    
    valid_mask = (target_speech >= 0) & (target_speech < max_speech_vocab)
    target_speech = target_speech.masked_fill(~valid_mask & (target_speech != IGNORE_ID), IGNORE_ID)
    
    loss_speech = F.cross_entropy(
        logits_speech.reshape(-1, logits_speech.size(-1)),
        target_speech.reshape(-1),
        ignore_index=IGNORE_ID
    )
    
    if torch.isnan(loss_text) or torch.isnan(loss_speech):
        logger.warning("⚠️ NaN loss detected!")
        return torch.tensor(0.0, device=device, requires_grad=True), \
               torch.tensor(0.0, device=device, requires_grad=True)
    
    return loss_text, loss_speech


def prepare_batch(model, batch, config):
    """Prepare batch - same as before"""
    device = config.device
    audios = batch["audio"]
    texts = batch["text"]
    
    batch_size = len(texts)
    
    audio_16k_list = []
    for audio in audios:
        audio_np = audio.numpy()
        audio_16k = librosa.resample(audio_np, orig_sr=model.sr, target_sr=S3_SR)
        audio_16k_list.append(audio_16k)
    
    s3_tokenizer = model.s3gen.tokenizer
    
    try:
        speech_tokens, speech_token_lens = s3_tokenizer.forward(audio_16k_list, max_len=1000)
        speech_tokens = speech_tokens.to(device)
    except Exception as e:
        logger.error(f"Tokenization failed: {e}")
        return None
    
    max_speech_vocab = model.t3.hp.speech_tokens_dict_size
    
    if speech_tokens.max() >= max_speech_vocab:
        logger.error(f"🚨 INVALID TOKEN IN BATCH: {speech_tokens.max()} >= {max_speech_vocab}")
        return None
    
    if isinstance(speech_token_lens, list):
        speech_token_lens = torch.tensor(speech_token_lens, device=device)
    else:
        speech_token_lens = speech_token_lens.to(device)
    
    sot_speech = model.t3.hp.start_speech_token
    eot_speech = model.t3.hp.stop_speech_token
    
    if sot_speech >= max_speech_vocab or eot_speech >= max_speech_vocab:
        logger.error(f"🚨 Invalid special tokens")
        return None
    
    speech_tokens = F.pad(speech_tokens, (1, 0), value=sot_speech)
    speech_tokens = F.pad(speech_tokens, (0, 1), value=eot_speech)
    speech_token_lens = speech_token_lens + 2
    
    text_tokens_list = []
    for text in texts:
        try:
            tokens = model.tokenizer.text_to_tokens(text, language_id=config.language_id)
            text_tokens_list.append(tokens.squeeze(0))
        except Exception as e:
            logger.error(f"Text tokenization failed: {e}")
            return None
    
    max_text_len = max(t.shape[0] for t in text_tokens_list)
    text_tokens = torch.zeros(batch_size, max_text_len, dtype=torch.long, device=device)
    text_token_lens = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    for i, t in enumerate(text_tokens_list):
        text_tokens[i, :t.shape[0]] = t.to(device)
        text_token_lens[i] = t.shape[0]
    
    sot_text = model.t3.hp.start_text_token
    eot_text = model.t3.hp.stop_text_token
    
    text_tokens = F.pad(text_tokens, (1, 0), value=sot_text)
    text_tokens = F.pad(text_tokens, (0, 1), value=eot_text)
    text_token_lens = text_token_lens + 2
    
    try:
        ref_audio = audio_16k_list[0]
        ref_audio = ref_audio[:model.ENC_COND_LEN]
        
        cond_speech_tokens = None
        if plen := model.t3.hp.speech_cond_prompt_len:
            cond_speech_tokens, _ = s3_tokenizer.forward([ref_audio], max_len=plen)
            cond_speech_tokens = cond_speech_tokens.to(device)
            
            if cond_speech_tokens.max() >= max_speech_vocab:
                logger.error(f"🚨 Invalid conditioning token")
                return None
            
            cond_speech_tokens = cond_speech_tokens.expand(batch_size, -1)
        
        ve_embed = torch.from_numpy(model.ve.embeds_from_wavs([ref_audio], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(device)
        ve_embed = ve_embed.expand(batch_size, -1)
        
        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=cond_speech_tokens,
            emotion_adv=0.5 * torch.ones(batch_size, 1, 1, device=device),
        )
    except Exception as e:
        logger.error(f"Conditioning preparation failed: {e}")
        return None
    
    return {
        "t3_cond": t3_cond,
        "text_tokens": text_tokens,
        "text_token_lens": text_token_lens,
        "speech_tokens": speech_tokens,
        "speech_token_lens": speech_token_lens,
    }


def train(config: FinetuneConfig):
    """Main training loop - PARTIAL FINE-TUNE"""
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    logger.info("Loading ChatterboxMultilingualTTS...")
    model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
    
    logger.info(f"📊 Model Configuration:")
    logger.info(f"  Speech vocab: {model.t3.hp.speech_tokens_dict_size} tokens")
    logger.info(f"  Text vocab: {model.t3.hp.text_tokens_dict_size} tokens")
    
    freeze_strategically(model, config)
    
    # ✅ NEW: Resume support for partial fine-tune
    resume_step = 0
    resume_epoch = 0
    
    if hasattr(config, 'resume_from_checkpoint') and config.resume_from_checkpoint:
        checkpoint_path = os.path.join(config.resume_from_checkpoint, "model.safetensors")
        
        if os.path.exists(checkpoint_path):
            logger.info(f"🔄 Resuming from: {config.resume_from_checkpoint}")
            
            try:
                # Load checkpoint weights
                checkpoint_weights = load_file(checkpoint_path, device="cpu")
                missing, unexpected = model.t3.load_state_dict(checkpoint_weights, strict=False)
                
                logger.info(f"✅ Loaded {len(checkpoint_weights)} parameters")
                logger.info(f"   Missing: {len(missing)}, Unexpected: {len(unexpected)}")
                
                # Try to extract step/epoch from checkpoint name
                checkpoint_name = os.path.basename(config.resume_from_checkpoint)
                
                if "checkpoint-" in checkpoint_name:
                    # checkpoint-2000 → step 2000
                    resume_step = int(checkpoint_name.split("-")[1])
                    logger.info(f"   Resuming from step: {resume_step}")
                    
                elif "epoch_" in checkpoint_name:
                    # epoch_0 → start of epoch 1
                    resume_epoch = int(checkpoint_name.split("_")[1]) + 1
                    logger.info(f"   Resuming from epoch: {resume_epoch}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load checkpoint: {e}")
                logger.warning(f"Starting fresh training instead")
        else:
            logger.warning(f"⚠️ Checkpoint not found: {checkpoint_path}")
            logger.warning(f"Starting fresh training instead")
    
    model.t3.to(config.device)
    model.s3gen.to(config.device)
    model.ve.to(config.device)
    
    logger.info("Loading dataset...")
    dataset = EgyptianArabicDataset(config, model)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    # ✅ Optimizer for ALL trainable parameters (not just LoRA!)
    optimizer = torch.optim.AdamW(
        [p for p in model.t3.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=0.01
    )
    
    total_steps = len(dataloader) * config.num_epochs // config.gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    logger.info(f"🚀 Starting PARTIAL FINE-TUNE for {config.num_epochs} epochs...")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    
    global_step = 0
    model.t3.train()
    skipped_batches = 0
    
    for epoch in range(config.num_epochs):
        epoch_loss = 0
        epoch_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                prepared = prepare_batch(model, batch, config)
                
                if prepared is None:
                    skipped_batches += 1
                    optimizer.zero_grad()
                    continue
                
                loss_text, loss_speech = compute_loss(model, prepared, config)
                
                if loss_speech.item() == 0 and loss_text.item() == 0:
                    skipped_batches += 1
                    optimizer.zero_grad()
                    continue
                
                loss = loss_speech + 0.1 * loss_text
                loss = loss / config.gradient_accumulation_steps
                loss.backward()
                
                epoch_loss += loss.item() * config.gradient_accumulation_steps
                epoch_batches += 1
                
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    total_norm = 0
                    for p in model.t3.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    total_norm = total_norm ** 0.5
                    
                    # ✅ Stricter gradient clipping for full fine-tune
                    torch.nn.utils.clip_grad_norm_(model.t3.parameters(), 0.5)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    pbar.set_postfix({
                        'loss': f"{epoch_loss/max(epoch_batches, 1):.4f}",
                        'grad': f"{total_norm:.2e}",
                        'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                        'step': global_step,
                        'skip': skipped_batches
                    })
                    
                    if global_step % config.save_steps == 0:
                        save_checkpoint(model, config, global_step)
                        
            except Exception as e:
                logger.error(f"❌ Error in batch {batch_idx}: {e}")
                optimizer.zero_grad()
                skipped_batches += 1
                continue
        
        avg_loss = epoch_loss / max(epoch_batches, 1)
        logger.info(f"✅ Epoch {epoch} completed | Avg Loss: {avg_loss:.4f} | Skipped: {skipped_batches}")
        
        # ✅ Save after each epoch (important for partial fine-tune!)
        save_checkpoint(model, config, global_step, epoch_final=epoch)
    
    save_checkpoint(model, config, global_step, final=True)
    logger.info(f"🎉 Training completed! Total skipped batches: {skipped_batches}")


def save_checkpoint(model, config, step, final=False, epoch_final=None):
    """Save model checkpoint - FULL T3 WEIGHTS (not just LoRA!)"""
    if final:
        save_path = os.path.join(config.output_dir, "final_model")
    elif epoch_final is not None:
        save_path = os.path.join(config.output_dir, f"epoch_{epoch_final}")
    else:
        save_path = os.path.join(config.output_dir, f"checkpoint-{step}")
    
    os.makedirs(save_path, exist_ok=True)
    
    # ✅ Save ALL trainable T3 parameters (not just LoRA adapters!)
    t3_state_dict = {k: v.cpu() for k, v in model.t3.state_dict().items() 
                     if any(p.requires_grad for p in model.t3.parameters())}
    
    save_file(t3_state_dict, os.path.join(save_path, "model.safetensors"))
    
    # Save config for reference
    import json
    training_config = {
        "freeze_text_encoder": config.freeze_text_encoder,
        "freeze_early_acoustic": config.freeze_early_acoustic,
        "unfreeze_prosody": config.unfreeze_prosody,
        "unfreeze_late_acoustic": config.unfreeze_late_acoustic,
        "learning_rate": config.learning_rate,
        "num_epochs": config.num_epochs,
    }
    
    with open(os.path.join(save_path, "training_config.json"), "w") as f:
        json.dump(training_config, f, indent=2)
    
    logger.info(f"💾 Saved checkpoint: {save_path} ({len(t3_state_dict)} parameters)")


if __name__ == "__main__":
    config = FinetuneConfig()
    train(config)
