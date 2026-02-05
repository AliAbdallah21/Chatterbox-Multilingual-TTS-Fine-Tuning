# 🎤 Egyptian Arabic TTS Fine-Tuning with Chatterbox

Fine-tuning Chatterbox Multilingual TTS for Egyptian Arabic dialect using strategic partial fine-tuning.

[![License](https://img.shields.io/badge/License-Chatterbox-blue.svg)](https://github.com/Chatterbox-TTS/chatterbox)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## 🎯 Overview

This project demonstrates the **first successful public fine-tuning** of the Chatterbox Multilingual TTS model for Egyptian Arabic dialect. Unlike traditional LoRA fine-tuning which only clones voice characteristics, this approach uses **strategic partial fine-tuning** to deeply learn Egyptian dialect features including:

- ✅ Egyptian pronunciation (ق → ء, ج → g)
- ✅ Natural Egyptian prosody and rhythm
- ✅ Colloquial Egyptian Arabic patterns
- ✅ High-quality voice preservation

### Key Achievement

**First open-source model** to successfully adapt a multilingual TTS system for Egyptian Arabic dialect while preserving base model capabilities.

## 🔊 Audio Samples

Listen to the results:
- [Sample 1](samples/sample_01.wav) - *"الكتاب ده عبارة عن حكايات على القهوة"*
- [Sample 2](samples/sample_02.wav) - *"انا بحب مصر جدا"*
- [Sample 3](samples/sample_06.wav) - *"اللغة العربية لغة جميلة وغنية بالتاريخ و الثقافة والادب و الحضارة العريقة زي الجمال بتاعها"*

## 📊 Dataset

- **Source**:online educational videos
- **Total Duration**: 120 hours of clean Egyptian Arabic
- **Samples**: 43,711 audio segments
- **Quality**: Filtered for English content, validated for audio quality
- **Speaker**: Single speaker

### Data Preparation Pipeline

The dataset was prepared with rigorous filtering:
1. Scraped 265 hours of raw video content
2. Removed English segments (3,000+ segments filtered)
3. Validated audio quality (duration, silence detection)
4. Normalized audio levels
5. Created clean metadata

## 🏗️ Architecture & Methodology

### Base Model
- **Model**: Chatterbox Multilingual TTS
- **Parameters**: 567M total (~500M in T3 module)
- **Languages**: 23 languages (preserved after fine-tuning)

### Fine-Tuning Strategy: Partial Fine-Tuning

Unlike LoRA (which only trains 5.5% of parameters in adapters), we use **strategic partial fine-tuning**:

```python
Frozen Components (Protected):
✅ Text encoder          → Protects phoneme knowledge
✅ Early acoustic layers → Protects low-level audio features
✅ S3Gen codec          → Protects audio generation quality
✅ Voice encoder        → Protects speaker embeddings

Trainable Components (Optimized):
🔥 Prosody predictors   → Learns Egyptian rhythm and intonation!
🔥 Late acoustic layers → Learns dialect-specific features + speaker

Result: 99.53% of T3 parameters trainable (strategically selected)
```

### Why This Works

| Method | Voice Quality | Dialect Learning | Stability | Training Time |
|--------|--------------|------------------|-----------|---------------|
| **LoRA** | ✅ Good | ❌ Minimal | ✅ Stable | 2 days (12 epochs) |
| **Full Fine-tune** | ✅ Excellent | ✅ Strong | ❌ Unstable | High risk |
| **Partial Fine-tune** | ✅ Excellent | ✅ **Deep** | ✅ Stable | 1.5 days (1 epoch) |

**Our approach achieves the best of both worlds!**

## 🚀 Quick Start

### Prerequisites

**Hardware Requirements:**
- NVIDIA GPU with 16GB+ VRAM (RTX 3090/4060Ti or better)
- 100GB+ free disk space
- CUDA 11.8+ and cuDNN

**Software Requirements:**
- Anaconda or Miniconda
- Python 3.11+
- Git

### Installation

#### Step 1: Set Up Environment

```bash
# Clone this repository
git clone https://github.com/AliAbdallah21/Chatterbox-Multilingual-TTS-Fine-Tuning.git
cd Chatterbox-Multilingual-TTS-Fine-Tuning

# Create conda environment
conda create -n chatterbox python=3.11
conda activate chatterbox
```

#### Step 2: Install PyTorch

```bash
# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### Step 3: Install Chatterbox

```bash
# Clone and install Chatterbox base model
cd ..
git clone https://github.com/Chatterbox-TTS/chatterbox.git
cd chatterbox
pip install -e .
cd ../Chatterbox-Multilingual-TTS-Fine-Tuning
```

#### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Data Preparation

#### Option A: Use Your Own Egyptian Arabic Data

Prepare your dataset in this structure:

```
raw_data/
├── playlist1/
│   ├── video1/
│   │   ├── segment_0001/
│   │   │   ├── audio.wav
│   │   │   └── transcript.txt
│   │   └── segment_0002/
│   │       ├── audio.wav
│   │       └── transcript.txt
│   └── video2/
└── playlist2/
```

Run the preparation script:

```bash
# Edit prepare_dahih_114h.py to set your SOURCE_DIR
# Then run:
python prepare_data.py
```

This creates:
```
output_dir/
├── wavs/
│   ├── audio_000001.wav
│   ├── audio_000002.wav
│   └── ...
├── metadata.csv          # filename|text format
└── metadata_full.csv     # Full metadata with durations
```

#### Option B: Prepare Your Dataset Manually

Create a dataset folder with:
- `wavs/` directory containing `.wav` files (24kHz recommended)
- `metadata.csv` with format: `filename|text`

Example:
```csv
audio_001.wav|النهارده الجو حلو قوي
audio_002.wav|ازيك يا باشا عامل ايه
```

### Training

#### Step 1: Configure Training

Edit `config.py`:

```python
@dataclass
class FinetuneConfig:
    # Update this path to your prepared dataset
    data_dir: str = "C:\\path\\to\\prepared_dataset"
    output_dir: str = "./output_data_egyptian_partial"
    
    # Training settings (recommended)
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    save_steps: int = 2000
    
    # Freezing strategy (critical for success!)
    freeze_text_encoder: bool = True
    freeze_early_acoustic: bool = True
    unfreeze_prosody: bool = True
    unfreeze_late_acoustic: bool = True
```

#### Step 2: Start Training

```bash
python train.py
```

**Expected timeline:**
- Epoch 0: ~13-14 hours
- Epoch 1: ~13-14 hours  
- Epoch 2: ~13-14 hours
- **Total: ~40 hours**

**Important**: Quality is achieved early! Checkpoint at step 2000 (73% of epoch 0) already produces excellent results.

#### Step 3: Monitor Training

Training logs will show:
```
Epoch 0:  73%|███| 16000/21856 [12:57:23<4:24:00, 2.70s/it, loss=4.5723, grad=2.20e+00, lr=8.90e-06, step=2000, skip=0]
💾 Saved checkpoint: ./output_dahih_egyptian_partial/checkpoint-2000 (292 parameters)
```

Checkpoints saved:
- `checkpoint-2000`, `checkpoint-4000`, etc. (every 2000 steps)
- `epoch_0`, `epoch_1`, `epoch_2` (after each epoch)
- `final_model` (after all epochs complete)

### Inference / Testing

Test any checkpoint:

```bash
# Test checkpoint from step 2000
python quick_partial_test.py --checkpoint checkpoint-2000

# Test after epoch 0 completes
python quick_partial_test.py --checkpoint epoch_0

# Test final model
python quick_partial_test.py --checkpoint final_model
```

Output audio saved in: `test_outputs/<checkpoint_name>/`

### Resume Training (Optional)

To resume from a checkpoint:

```python
# In config.py, add:
resume_from_checkpoint: str = "./output_data_egyptian_partial/checkpoint-2000"
```

Then run `python train.py` again.

## 📈 Training Results & Insights

### Loss Progression

```
Initial:      ~6.0  (not logged due to gradient accumulation)
Step 2000:    4.57  (73% of epoch 0) ← Already excellent quality!
Epoch 0 end:  4.50
Epoch 1 end:  4.25
Final (epoch 2): ~4.0
```

### Key Findings

1. **LoRA is insufficient for dialect learning**
   - LoRA (12 epochs): Loss 4.70 → 4.21
   - Result: Voice cloning only, no Egyptian dialect features
   
2. **Partial fine-tuning learns dialects deeply**
   - Partial (1 epoch): Loss 6.0 → 4.50
   - Result: Full Egyptian pronunciation, rhythm, and naturalness
   
3. **Quality peaks early**
   - Checkpoint-2000 (73% epoch 0): Already produces near-perfect Egyptian speech
   - Further training: Marginal improvements, risk of overfitting

4. **Loss ≠ Audio Quality in TTS**
   - Lower loss doesn't guarantee better audio
   - Always evaluate by listening, not just metrics

### Comparison: LoRA vs Partial Fine-Tune

| Aspect | LoRA (Previous) | Partial Fine-Tune (This Work) |
|--------|----------------|------------------------------|
| **Egyptian ق** | ❌ Sounds like MSA | ✅ Perfect ء sound |
| **Egyptian ج** | ❌ Sounds like MSA | ✅ Perfect g sound |
| **Rhythm** | ❌ Formal/MSA-like | ✅ Natural Egyptian |
| **Naturalness** | ⚠️ Slightly robotic | ✅ Very natural |
| **Training time** | 48 hours (12 epochs) | 14 hours (1 epoch) |
| **Quality** | 5/10 | 10/10 🎯 |

## 🔧 Advanced Configuration

### Hyperparameter Tuning

```python
# Conservative (safer, slower convergence)
learning_rate: float = 5e-6
num_epochs: int = 5

# Aggressive (faster, higher risk)
learning_rate: float = 2e-5
num_epochs: int = 2

# Recommended (balanced)
learning_rate: float = 1e-5  # Sweet spot!
num_epochs: int = 3
```

### Adjusting Trainable Layers

To freeze more aggressively (if model diverges):

```python
freeze_text_encoder: bool = True
freeze_early_acoustic: bool = True
unfreeze_prosody: bool = True
unfreeze_late_acoustic: bool = False  # Freeze late acoustic too
```

To train more (if quality is insufficient):

```python
freeze_text_encoder: bool = False  # Unfreeze text encoder
freeze_early_acoustic: bool = True
unfreeze_prosody: bool = True
unfreeze_late_acoustic: bool = True
```

## 📁 Repository Structure

```
egyptian-finetune/
├── config.py                    # Training configuration
├── train.py                     # Main training script with partial fine-tuning
├── dataset.py                   # PyTorch dataset loader
├── prepare_data.py              # Dataset preparation and filtering
├── quick_partial_test.py        # Inference testing script
├── requirements.txt             # Python dependencies
├── samples/                     # Audio samples (for demonstration)
│   ├── sample_01.wav
│   ├── sample_02.wav
│   └── sample_06.wav
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 💡 Technical Insights

### Why Partial Fine-Tuning Succeeds

**The Problem with LoRA:**
- Only adapts attention projection matrices
- Small capacity (5.5% of parameters)
- Cannot learn deep prosodic and phonetic changes
- Result: Voice cloning without dialect learning

**The Problem with Full Fine-Tuning:**
- Trains all 567M parameters
- Catastrophic forgetting of other languages
- High risk of overfitting on single speaker
- Unstable training

**Our Solution: Strategic Partial Fine-Tuning:**
- Freeze components that preserve base knowledge
- Unfreeze components responsible for prosody and speaker characteristics
- 99.53% trainable but carefully selected
- Deep dialect learning without catastrophic forgetting

### Freezing Strategy Rationale

| Component | Status | Reason |
|-----------|--------|--------|
| Text Encoder | ❄️ Frozen | Protects grapheme-to-phoneme mappings |
| Early Acoustic | ❄️ Frozen | Protects low-level audio features (formants, harmonics) |
| **Prosody Predictor** | 🔥 **Trainable** | **Learns Egyptian rhythm, stress, intonation** |
| **Late Acoustic** | 🔥 **Trainable** | **Learns dialect phonetics + speaker timbre** |
| S3Gen Codec | ❄️ Frozen | Protects audio synthesis quality |
| Voice Encoder | ❄️ Frozen | Protects speaker embedding space |

### Preventing Overfitting

1. **Low learning rate** (1e-5): Prevents aggressive updates
2. **Few epochs** (3 max): Stops before memorization
3. **Strategic freezing**: Limits parameter space
4. **Early stopping**: Best checkpoint often at 70-80% of epoch 0
5. **Gradient clipping** (0.5): Prevents instabilities

## 🎓 Citation

If you use this work in your research or projects, please cite:

```bibtex
@misc{abdallah2026egyptian,
  title={Egyptian Arabic TTS Fine-Tuning with Chatterbox: A Partial Fine-Tuning Approach},
  author={Ali Abdallah},
  year={2026},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\\url{https://github.com/AliAbdallah21/Chatterbox-Multilingual-TTS-Fine-Tuning}}
}
```

## 📝 License

This project follows the Chatterbox model license. Please refer to the [Chatterbox repository](https://github.com/Chatterbox-TTS/chatterbox) for license details.

**Important**: The fine-tuned model weights are derivative works and subject to the original Chatterbox license terms.

## 🙏 Acknowledgments

- **Chatterbox Team** for developing the exceptional multilingual TTS architecture
- **Egyptian Arabic NLP Community** for inspiration and support
- **Open source community** for tools and frameworks (PyTorch, Hugging Face, etc.)

## 🤝 Contributing

Contributions are welcome! Here are areas for improvement:

### Immediate Opportunities
- [ ] Add more Egyptian speakers for multi-speaker capability
- [ ] Create evaluation metrics for dialect quality assessment
- [ ] Optimize training efficiency (mixed precision, gradient checkpointing)
- [ ] Add support for other Egyptian regional variations

### Long-term Goals
- [ ] Extend to other Arabic dialects (Levantine, Gulf, Maghrebi)
- [ ] Create dialect classifier for automatic dialect detection
- [ ] Build web interface for easy inference
- [ ] Develop real-time streaming TTS

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🐛 Known Issues & Troubleshooting

### Issue: GPU Out of Memory

**Solution**: Reduce batch size in `config.py`:
```python
batch_size: int = 1  # Instead of 2
```

### Issue: Training loss starts very high (>6.0)

**Expected behavior**: Initial loss ~6.0 is normal for cross-entropy on 8,194 token vocabulary.

### Issue: Checkpoint sounds robotic

**Solution**: This checkpoint may be overfitted. Try an earlier checkpoint (e.g., checkpoint-2000 often sounds better than final_model).

### Issue: Model doesn't sound Egyptian

**Possible causes**:
1. Dataset not truly Egyptian (check for MSA contamination)
2. Training stopped too early (before step 2000)
3. Learning rate too low (increase to 1e-5)

## 📧 Contact & Support

- **GitHub**: [@AliAbdallah21](https://github.com/AliAbdallah21)
- **Repository**: [Chatterbox-Multilingual-TTS-Fine-Tuning](https://github.com/AliAbdallah21/Chatterbox-Multilingual-TTS-Fine-Tuning)
- **Issues**: [Report bugs or request features](https://github.com/AliAbdallah21/Chatterbox-Multilingual-TTS-Fine-Tuning/issues)
- **Email**: [E-mail](aliabdalla2110@gmail.com)

For collaboration or commercial inquiries, please open an issue or reach out via GitHub or Email.

## 🌟 Star History

If you find this project useful, please consider starring the repository! It helps others discover this work.

---

## 🎯 Impact & Future Work

### Research Impact

This work demonstrates:
1. **First successful public adaptation** of a multilingual TTS for Egyptian dialect
2. **Novel partial fine-tuning strategy** that balances dialect learning with stability
3. **Practical methodology** for low-resource dialect adaptation
4. **Open-source contribution** to Arabic NLP/TTS community

### Applications

- 🎙️ Egyptian voice assistants
- 📚 Audiobook narration in Egyptian Arabic
- 🎬 Dubbing and localization
- 🎓 Educational content in native dialect
- ♿ Accessibility tools for Egyptian Arabic speakers
- 🎮 Game character voices
- 📱 Virtual assistant applications

### Next Steps

We are working on:
1. Multi-speaker Egyptian TTS (adding 5-10 more Egyptian speakers)
2. Emotional control (happy, sad, excited, neutral)
3. Real-time inference optimization
4. Cross-dialect adaptation (using this as base for other Arabic dialects)

---

**🇪🇬 First open-source, high-quality Egyptian Arabic TTS model! 🎤**

**Made with ❤️ for the Egyptian and Arabic-speaking communities**