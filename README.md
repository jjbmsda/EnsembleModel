# LibriSpeech Classification with ResNet, DenseNet, and Ensemble Models

This repository contains a PyTorch implementation for speaker classification using the **LibriSpeech** dataset. 
It leverages ResNet, DenseNet, and an Ensemble model combining both architectures for enhanced performance.

## Features
- **MFCC-based Feature Extraction**: Uses torchaudio's `MFCC` transformation for audio preprocessing.
- **Model Architectures**:
  - **ResNet18**: Lightweight convolutional neural network for efficient training.
  - **DenseNet121**: Compact and accurate model with dense connections.
  - **Ensemble Model**: Combines ResNet18 and DenseNet121 outputs for better accuracy.
- **Mixed Precision Training**: Reduces memory usage and speeds up training using `torch.cuda.amp`.
- **Dynamic Label Mapping**: Automatically maps dataset labels to indices, preventing label mismatches.

## Dataset
This project uses the **LibriSpeech ASR Corpus**, a large-scale corpus of read English speech:
- Training data: `train-clean-100`
- Testing data: `test-clean`

The dataset is automatically downloaded and preprocessed using `torchaudio.datasets.LIBRISPEECH`.

## Prerequisites
### Libraries
- `torch`
- `torchaudio`
- `torchvision`
- `scikit-learn`
- `numpy`

### Installation
```bash
pip install torch torchaudio torchvision scikit-learn numpy
```

## Usage
### Training and Evaluation
1. Clone this repository:

```bash
git clone https://github.com/yourusername/librispeech-classification.git
cd librispeech-classification
```
2. Run the main script:

```bash
python main.py
```

### Key Parameters
- **Batch Size**: Default is 8 for optimal GPU memory usage.
- **Epochs**: Default is set to 5 but can be adjusted.
- **Model Input Dimensions**: Automatically resized to 50x50 using a custom resizing function.

## Models
### ResNet18
A lightweight convolutional neural network with minimal memory footprint.

### DenseNet121
A densely connected neural network with improved gradient flow and compact representation.

### Ensemble Model
Combines the outputs of ResNet18 and DenseNet121 by averaging their predictions:

```python
resnet_out = resnet(x)
densenet_out = densenet(x)
ensemble_out = (resnet_out + densenet_out) / 2
```

## Results
- Loss and accuracy are displayed for each epoch.
- The Ensemble Model aims to improve classification performance compared to individual models.

## Contributing
Pull requests and feature suggestions are welcome. Please fork this repository and submit your contributions.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
- LibriSpeech Dataset: https://www.openslr.org/12
- PyTorch: https://pytorch.org
- Torchaudio: https://pytorch.org/audio

