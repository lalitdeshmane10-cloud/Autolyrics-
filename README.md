# AutoLyrics - Whisper-Based Singing Lyrics Transcription

**Fine-tune OpenAI's Whisper model for accurate song lyric transcription using Low-Rank Adaptation (LoRA)**

*By: Ayush Bansal, Bhumika, Lalit Deshmane*

---

##  Project Overview

AutoLyrics is a **parameter-efficient fine-tuning pipeline** that adapts OpenAI's Whisper-small model specifically for transcribing singing vocals and song lyrics. Using Low-Rank Adaptation (LoRA), the project achieves significant accuracy improvements while maintaining minimal computational overhead through 8-bit quantization and gradient checkpointing.

**Key Capability**: Compare baseline Whisper predictions against LoRA-fine-tuned predictions through an interactive Gradio web interface.

---

## Dataset Structure

### Directory Contents
```
merged_data/
├── Autolyrics (1) (1).ipynb       # Main training & deployment notebook
├── train (2).jsonl                # Training dataset (~150 samples)
├── val (1).jsonl                  # Validation dataset
├── test (1).jsonl                 # Test dataset (12 samples)
├── audio/                         # Audio chunks directory (500+ WAV files)
└── README.md                      # This file
```

### JSONL Data Format
Each line in the dataset files contains:
```json
{
  "id": "ed_divide_011_chunk010",
  "audio_filepath": "audio/ed_divide_011_chunk010.wav",
  "duration": 21.63,
  "text": "tell me that you love me too tell me that you love me too",
  "start": 258.93,
  "end": 280.56,
  "track_name": "How Would You Feel (Paean)",
  "artist_name": "Ed Sheeran"
}
```

### Audio Files
500+ audio chunks in WAV format at **16kHz mono** from:
- **Ed Sheeran** - "Divide" album songs
- **Olivia Rodrigo** - "Sour" album songs  
- **Taylor Swift** - "Fearless (Taylor's Version)", "Folklore (Deluxe)", "Red (Taylor's Version)", "Speak Now (Taylor's Version)"

**Source Dataset**: [gmenon/slt-lyrics-audio on Hugging Face](https://huggingface.co/datasets/gmenon/slt-lyrics-audio)

---
## Notebook Workflow

The main notebook (`Autolyrics (1) (1).ipynb`) implements 7 sequential stages:

### Stage 1: Environment Setup
**Dependencies installed:**
- `torch`, `torchaudio` - Deep learning framework
- `transformers` - Hugging Face model library
- `datasets` - Data loading utilities
- `peft` - Parameter-Efficient Fine-Tuning
- `bitsandbytes` - 8-bit quantization
- `jiwer` - Evaluation metrics (WER, CER)
- `gradio` - Web UI framework
- `accelerate` - Distributed training

### Stage 2: Configuration
Core hyperparameters:
```python
MODEL_NAME = "openai/whisper-small"
DATASET_NAME = "gmenon/slt-lyrics-audio"
OUTPUT_DIR = "./whisper-lora-autolyrics"

BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-4
MAX_STEPS = 100
```

### Stage 3: Data Pipeline
**Processing steps:**
1. Load dataset from Hugging Face Hub
2. Resample all audio to 16kHz (Whisper's required sampling rate)
3. Extract mel-spectrogram features (80 mel bins)
4. Tokenize lyrics text using WhisperProcessor
5. Implement `DataCollatorSpeechSeq2SeqWithPadding` for dynamic padding

**Custom Data Collator**: Handles variable-length audio sequences by padding input features and labels while preserving attention masks.

### Stage 4: Model Initialization & PEFT
**Memory Optimization:**
- **8-bit Quantization**: Reduces model size by 4x (only on GPU)
- **LoRA Configuration**:
  - Rank (r) = 32
  - Alpha = 64
  - Target modules: `q_proj`, `v_proj` (attention layers)
  - Dropout = 0.05
  - Bias = "none"
- **Result**: ~99% fewer trainable parameters vs. full fine-tuning

### Stage 5: Model Training & Serialization
**Training Configuration:**
- Framework: Seq2Seq Trainer
- Optimization: Gradient checkpointing + FP16 mixed precision
- Warmup steps: 10
- Checkpoint frequency: Every 50 steps
- Evaluation frequency: Every 50 steps
- Loss: Cross-entropy (task-specific)
- Best model selection: Based on minimum WER

**Output Artifacts Saved:**
```
whisper-lora-autolyrics/
├── adapter_config.json         # LoRA architecture config
├── adapter_model.bin           # LoRA weights
├── config.json                 # Whisper model config
├── generation_config.json      # Decoding parameters
└── preprocessor_config.json    # Feature extractor config
```

### Stage 6: Performance Evaluation
**Metrics Computed:**
- **Word Error Rate (WER)**: Percentage of words incorrectly transcribed
- **Character Error Rate (CER)**: Percentage of characters incorrectly transcribed
- **Relative Improvement**: `(BaseWER - LoRAWER) / BaseWER × 100%`

**Evaluation Process:**
1. Generate predictions from baseline Whisper model
2. Generate predictions from LoRA-fine-tuned model
3. Compare against ground-truth labels
4. Print side-by-side metrics

### Stage 7: Interactive Deployment - Gradio Web Interface
**Web Application Features:**
- **Audio Input**: Upload MP3, WAV, OGG, FLAC, M4A, MPEG, or MP4 files
- **Parallel Inference**: 
  - LoRA fine-tuned model (top output)
  - Baseline Whisper-small (bottom output)
- **Beam Search**: 3-beam decoding for better accuracy
- **Copy Buttons**: One-click transcript copying
- **Dark Theme**: Modern UI with gradient title

**Technical Backend:**
- Real-time mono downmixing & 16kHz resampling
- Batch processing with BeamSearchDecoder (width=3)
- No-grad inference for efficiency
- Cache enabled for faster generation

---

## Quick Start Guide

### Prerequisites & Installation
```bash
# Clone/download the notebook
cd merged_data/

# Install dependencies (all at once or via pip)
pip install torch torchaudio transformers datasets peft bitsandbytes jiwer gradio accelerate

# Optional: Install additional audio processing
pip install librosa
```

### Running the Notebook
1. **Open in Jupyter or Google Colab**:
   ```bash
   jupyter notebook "Autolyrics (1) (1).ipynb"
   ```
   OR upload directly to Google Colab

2. **Execute cells sequentially** (top-to-bottom)
   - Stage 1-2: ~2-3 minutes (setup & config)
   - Stage 3: ~5 minutes (data loading & preprocessing)
   - Stage 4: ~3 minutes (model initialization)
   - Stage 5: ~30-60 minutes (training on GPU)
   - Stage 6: ~3 minutes (evaluation)
   - Stage 7: ~2 minutes (Gradio UI launch)

3. **GPU Requirement**: 
   - Recommended: NVIDIA GPU with 8GB+ VRAM (e.g., T4, V100, A100)
   - Fallback: CPU mode supported but ~10x slower

### Using the Web Interface
- After Stage 7 completes, a public Gradio link appears in the notebook output
- Share this link to enable others to test the model
- Upload audio → Click Submit → View transcriptions

---

## Expected Performance

| Metric | Baseline (Whisper-small) | LoRA Fine-Tuned | Improvement |
|--------|--------------------------|-----------------|-------------|
| WER    | ~15-20%                  | ~5-10%          | 40-50%      |
| CER    | ~8-12%                   | ~3-6%           | 40-50%      |
| Latency| ~0.5s per 10s audio      | ~0.5s per 10s audio | Same (inference-time efficient) |

---
##  Technical Deep Dive

### Model Architecture
- **Base**: OpenAI Whisper-small (244M parameters)
- **Type**: Seq2Seq Transformer (48-layer encoder + 12-layer decoder)
- **Input**: Mel-spectrogram (80 bands, ~3000 timesteps for 30s audio)
- **Output**: Token sequence decoded to text

### LoRA Adaptation
- **Approach**: Inject low-rank matrices into attention layers
- **Formula**: `Output = Wx + BAx` where B, A are trainable, W is frozen
- **Advantage**: 0.5M parameters vs. 244M (0.2% overhead)

### Training Specifics
- **Loss Function**: Cross-entropy (causal language modeling)
- **Optimizer**: AdamW (momentum, weight decay)
- **Scheduler**: Linear warmup then decay
- **Gradient Accumulation**: Simulate batch size 8 with effective batch 4
- **Precision**: FP16 (except model quantization in 8-bit)

### Inference Optimization
- **Quantization During Inference**: No (LoRA weights are FP32)
- **KV-Cache**: Enabled for fast sequential generation
- **Beam Width**: 3 (balance between speed & accuracy)
- **Max Tokens**: 225 (sufficient for ~30s audio)

---

## File Descriptions

| File | Purpose |
|------|---------|
| `Autolyrics (1) (1).ipynb` | Main notebook with all 7 stages |
| `train (2).jsonl` | 150 training samples |
| `val (1).jsonl` | Validation dataset |
| `test (1).jsonl` | 12 test samples for evaluation |
| `audio/` | 500+ audio chunks (organized by artist) |

---

##  Customization & Extension

### Adjust Training Parameters
Edit Stage 2 (Configuration):
```python
BATCH_SIZE = 8              # Increase for more stability
LEARNING_RATE = 5e-5        # Lower for gentler updates
MAX_STEPS = 500             # Longer training
GRADIENT_ACCUMULATION_STEPS = 4  # Simulate larger batches
```

### Change Model Size
```python
MODEL_NAME = "openai/whisper-base"      # Larger (73M params)
MODEL_NAME = "openai/whisper-medium"    # Even larger (769M params)
```

### Add Custom Validation
Stage 6 currently evaluates on 5 samples; modify:
```python
for sample in eval_data.select(range(50)):  # Evaluate 50 samples
```

---

##  Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of Memory | Reduce `BATCH_SIZE` to 2, enable quantization, or use smaller model |
| Slow Training | Enable GPU (CUDA), verify PyTorch sees GPU via `torch.cuda.is_available()` |
| Poor Transcriptions | Collect more training data, increase `MAX_STEPS`, tune `LEARNING_RATE` |
| Gradio Connection Error | Check firewall, ensure port access, or use `share=False` and access locally |

---

##  References & Resources

- **Whisper Paper**: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- **LoRA Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **Dataset**: [gmenon/slt-lyrics-audio](https://huggingface.co/datasets/gmenon/slt-lyrics-audio) - Singing Lyrics Transcription Audio Dataset
- **Hugging Face Docs**:
  - [Transformers](https://huggingface.co/docs/transformers)
  - [PEFT Library](https://huggingface.co/docs/peft)
  - [Datasets](https://huggingface.co/docs/datasets)
- **Evaluation**:
  - [jiwer](https://github.com/jitsi/jiwer) - WER/CER metrics
  - [Gradio](https://gradio.app) - Web UI framework

---

## Citation

If you use this project, cite as:
```bibtex
@misc{autolyrics2024,
  title={AutoLyrics: LoRA Fine-tuned Whisper for Song Lyric Transcription},
  authors={Bansal, Ayush and Bhumika and Deshmane, Lalit},
  year={2024}
}
```

---

## License

This project is built upon:
- **Whisper**: [OpenAI](https://github.com/openai/whisper) (MIT License)
- **Transformers**: [Hugging Face](https://github.com/huggingface/transformers) (Apache 2.0)
- **Datasets**: [Hugging Face](https://github.com/huggingface/datasets) (Apache 2.0)

Refer to individual component licenses for usage restrictions.
