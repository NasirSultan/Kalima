# Kalima (كلمة)

**Fine-tuned Whisper Tiny for Arabic Automatic Speech Recognition**

Kalima ("word" in Arabic) fine-tunes OpenAI's Whisper Tiny on Arabic speech. The goal is to make it accurate enough for real transcription work while keeping the speed and small size of Tiny.

---

## Results

Evaluated on the same 1,000-sample test set of [CUAIStudents/Ar-ASR](https://huggingface.co/datasets/CUAIStudents/Ar-ASR).

| Model | WER | CER | Word Accuracy | Char Accuracy |
|---|---|---|---|---|
| Whisper Tiny (base) | 70.1% | 29.5% | 29.9% | 70.5% |
| Kalima v1 | 14.7% | 4.8% | 85.3% | 95.2% |
| **Kalima v2** | **14.3%** | **4.2%** | **85.7%** | **95.8%** |

**Kalima v2 vs base:** WER improved by 79.6%, CER improved by 85.8%.

> Greedy decoding gives the best results. Beam search (5 beams) reduced accuracy for this model size (15.5% WER).

---

## Highlights

- **Before and after evaluation** on an identical test set, with per-sample transcripts saved for review.
- **Fully resumable training.** Checkpoints, partial evaluations and progress markers sync to the Hugging Face Hub, so a session timeout or kernel restart loses no progress.
- **Arabic-aware evaluation.** Metrics are computed after removing diacritics (tashkeel) and normalizing alif, ya and ta marbuta forms.
- **Best-checkpoint selection.** v2 tracks WER on a held-out validation set and keeps the best checkpoint, not the last one.
- **Error analysis** that groups errors into substitutions, deletions, insertions, 1-letter near-misses, and suspicious references.

---

## Dataset

- **Source:** [CUAIStudents/Ar-ASR](https://huggingface.co/datasets/CUAIStudents/Ar-ASR)
- **Language:** Modern Standard Arabic
- **Splits used:** train (about 33.6k samples), test (1,000 samples)
- **Validation:** 500 samples held out from train with a fixed seed, used in v2

---

## Training

| | v1 | v2 |
|---|---|---|
| Starting point | `openai/whisper-tiny` | Kalima v1 |
| Steps | 4,000 | 3,000 |
| Learning rate | 5e-5 | 1e-5 |
| Warmup | 500 | 200 |
| Batch | 16 × 2 (gradient accumulation) | 16 × 2 (gradient accumulation) |
| Precision | fp16 | fp16 |
| Checkpoint selection | Last | Best validation WER |

**Hardware:** Kaggle, single NVIDIA T4 GPU.

---

## Error analysis (v2)

| Finding | Detail |
|---|---|
| Error types | 83.5% substitutions, 10.3% deletions, 6.2% insertions |
| 1-letter near-misses | 26% of all errors. Counting them as correct gives 92.1% word accuracy. |
| Label noise | Only 0.4% of references are suspicious, so the dataset is clean |
| Weak areas | Quranic recitation, very short clips, and word segmentation |

---

## Usage

```python
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

repo = "Nasir6/whisper-tiny-ar-asr"
subfolder = "v2_final_model"

processor = WhisperProcessor.from_pretrained(repo, subfolder=subfolder)
model = WhisperForConditionalGeneration.from_pretrained(repo, subfolder=subfolder)

audio, _ = librosa.load("sample.wav", sr=16000)
features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features

ids = model.generate(features, language="ar", task="transcribe")
print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```

---

## Pipeline

```
Ar-ASR dataset
      ↓
Whisper Tiny baseline test → save transcripts + WER/CER
      ↓
Fine-tune (v1) → resumable, synced to Hugging Face Hub
      ↓
Continue training (v2) → validation + best checkpoint
      ↓
Test on the same test set → save transcripts + WER/CER
      ↓
Compare before vs after + error analysis
```

---

## Roadmap

- [x] Baseline evaluation of Whisper Tiny
- [x] v1 fine-tuning
- [x] v2 continued training with best-checkpoint selection
- [x] Error analysis
- [ ] **v3:** SpecAugment, cosine LR schedule, and added Quranic recitation data
- [ ] Larger base model (Whisper Small) for higher accuracy
- [ ] Faster inference with faster-whisper (CTranslate2)
- [ ] Gulf Arabic dialect support

---

## Tech stack

Python · PyTorch · Hugging Face Transformers · Datasets · Hugging Face Hub · jiwer · Kaggle

---

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper)
- [CUAIStudents/Ar-ASR](https://huggingface.co/datasets/CUAIStudents/Ar-ASR) dataset
- Hugging Face Transformers

---

## Author

**Nasir Sultan**, AI Engineer
[GitHub](https://github.com/NasirSultan) · [Hugging Face](https://huggingface.co/Nasir6) · [LinkedIn](https://linkedin.com/in/nasir-sultan-848762282)

---

## License

MIT
