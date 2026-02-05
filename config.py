from dataclasses import dataclass
from typing import Tuple

@dataclass
class FinetuneConfig:
    data_dir: str = "C:\\Users\\11\\chatterbox-finetuning\\MyTTSDataset_Dahih_114h"
    output_dir: str = "./output_dahih_egyptian_partial"
    
    # ✅ RESUME SUPPORT - Set this to resume training!
    # resume_from_checkpoint: str = "./output_dahih_egyptian_partial/checkpoint-2000"
    
    device: str = "cuda"
    language_id: str = "ar"
    
    freeze_text_encoder: bool = True
    freeze_early_acoustic: bool = True
    unfreeze_prosody: bool = True
    unfreeze_late_acoustic: bool = True
    
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    warmup_steps: int = 300
    save_steps: int = 2000
    logging_steps: int = 10
    
    sample_rate: int = 24000