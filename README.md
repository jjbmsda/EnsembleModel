# Speaker Classification with ResNet & DenseNet Ensemble (LibriSpeech)

This repository contains a **PyTorch-based speaker classification** project using the **LibriSpeech dev-clean** dataset.
The goal is to compare single-model performance (ResNet18, DenseNet121) and demonstrate **logit-level ensemble** improvements.

The code is designed to run **reliably on Google Colab** and local machines, and it explicitly avoids common issues related to
`torchaudio` backends and `torchcodec` mismatches.

---

## 1. Features

- Closed-set **speaker classification** (not ASR)
- Dataset: **LibriSpeech dev-clean**
- Models:
  - ResNet18
  - DenseNet121
- Evaluation:
  - Accuracy
  - Macro F1-score
  - Top-k Accuracy (Top-1 / Top-3 / Top-5)
- Ensemble:
  - Logit averaging at evaluation time
- AMP (Mixed Precision) support on GPU
- Safe handling of unseen speakers and empty batches
- Google Colab–friendly setup

---

## 2. Dataset

**LibriSpeech – dev-clean**

- Size: ~322 MB
- Clean speech recordings
- Multiple speakers

The dataset is **split internally** into:

- Train: 80%
- Validation: 10%
- Test: 10%

> ⚠️ `test-clean` is **not used** because most speakers do not overlap with training speakers.
> This project focuses on **closed-set speaker identification**.

---

## 3. Project Structure

```text
project_root/
├── README.md
├── train.py   # or notebook with main()
└── data/
    └── LibriSpeech/
        └── dev-clean/
            └── speaker_id/
                └── chapter_id/
                    └── *.flac
```

The dataset is downloaded automatically if it does not exist.

---

## 4. Environment Setup

### Requirements

- Python 3.10+
- torch
- torchvision
- torchaudio
- torchcodec
- soundfile
- scikit-learn

### Installation

```bash
pip install torch torchvision torchaudio
pip install torchcodec soundfile scikit-learn
```

> Notes:
> - Recent versions of `torchaudio` require **torchcodec**.
> - `torchaudio.set_audio_backend()` is deprecated and **should not be used**.

---

## 5. How to Run

### Local or Colab

```bash
python train.py
```

or run all cells in the notebook.

The script will:

1. Download `dev-clean` (if missing)
2. Convert audio to MelSpectrogram + dB scale
3. Split into train / validation / test
4. Train ResNet18 and DenseNet121
5. Evaluate validation performance
6. Evaluate test performance
7. Report ensemble results

---

## 6. Training Pipeline

1. Load waveform
2. Convert to mono
3. Resample to 16 kHz
4. Compute MelSpectrogram
5. Convert to decibel scale
6. Pad or trim time axis
7. Map speaker ID → class index
8. Train with CrossEntropyLoss

ResNet and DenseNet are trained **independently**.
Ensemble results are computed **only during evaluation**.

---

## 7. Evaluation Metrics

Reported metrics:

- Accuracy
- Macro F1-score
- Top-1 Accuracy
- Top-3 Accuracy
- Top-5 Accuracy

### Example Output

```text
Epoch 1/1
ResNet   loss=1.23  val_acc=0.83  macroF1=0.80
DenseNet loss=1.43  val_acc=0.75  macroF1=0.71
Ensemble           val_acc=0.89  macroF1=0.86

TEST RESULTS
ResNet   acc=0.82  macroF1=0.79  top3=0.97
DenseNet acc=0.75  macroF1=0.72  top3=0.98
Ensemble acc=0.88  macroF1=0.85  top3=1.00
```

---

## 8. Performance Tips

- Reduce `target_frames` (e.g. 256 → 160) for faster training
- Increase `batch_size` if GPU memory allows
- Use 1–3 epochs on free Google Colab
- Cache MelSpectrogram features for large-scale experiments

---

## 9. Common Issues

### NoneType object is not subscriptable

Occurs when the evaluation dataset is empty.

✔ Solution: Use **dev-clean internal split only**.

---

### torchcodec ImportError

```text
ImportError: TorchCodec is required
```

✔ Solution:

```bash
pip install torchcodec
```

---

### Very slow training

✔ Solutions:
- Reduce dataset size
- Reduce `target_frames`
- Use GPU runtime

---

## 10. Future Improvements

- Speaker-balanced sampling
- Metric learning (ArcFace, AM-Softmax)
- Speaker verification (EER evaluation)
- Feature caching pipeline
- Larger backbone models

---

## 11. License

This project is for **research and educational purposes**.

LibriSpeech is distributed under its own license.

---

## 12. Contact

Issues and improvement suggestions are welcome.

