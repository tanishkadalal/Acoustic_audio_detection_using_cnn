# Environmental_audio_detection_using_cnn

Sure! Here's a clean, professional `README.md` for your ESC-50 CNN Audio Detection project:

---

# ESC-50 Audio Classification using CNN

This project focuses on environmental sound classification using the ESC-50 dataset. A CNN-based deep learning model is trained on Mel-Spectrograms extracted from audio files to classify various environmental audio classes.

## 📁 Dataset

- **Source**: [ESC-50: Dataset for Environmental Sound Classification](https://github.com/karoldvl/ESC-50)
- **Downloaded via**: [`kagglehub`](https://github.com/KaggleHub)
- **Format**: WAV files, 50 classes, 5-second recordings.

## 🧠 Objective

To build a convolutional neural network (CNN) model that classifies environmental audio signals by learning from their Mel-Spectrogram representations.

## ⚙️ Project Setup

1. Install dependencies:
   ```bash
   pip install tensorflow-io[tensorflow]
   ```

2. Dataset download:
   ```python
   import kagglehub
   path = kagglehub.dataset_download("mmoreaux/environmental-sound-classification-50")
   ```

## 🔉 Audio Preprocessing

- **Audio Loading**: TensorFlow I/O is used to load `.wav` files.
- **Spectrogram Generation**: Raw audio is converted into Mel-Spectrograms using:
  - FFT + Mel-scaling
  - Resizing to (128, 86)
- **Normalization**: Each Mel-Spectrogram is standardized by subtracting mean and dividing by standard deviation.

## 🎛️ Data Augmentation Techniques

To increase robustness, we apply:
- **Time Stretching** (`rate=0.8`)
- **Pitch Shifting** (`+2 semitones`)
- **Noise Injection**

These augmentations simulate various real-world distortions in environmental audio.

## 📊 Model Architecture

A custom CNN with residual connections:
- **Conv2D** blocks with increasing filter sizes (32 → 64 → 128)
- **Residual Shortcut**: Adds a parallel downsampled input to the output
- **Flattened Output**: Feeds into classification layer (not shown in snippet)

```text
Input → Conv2D → MaxPool → Conv2D → MaxPool → Conv2D → MaxPool
    ↘------------------------ Residual Conv2D Path ----------------↗
                          → Add() → Flatten → Dense (Classification)
```

## 📈 Visualizations

- **Waveform** plots
- **Mel-Spectrograms** (log-scale)
- **Before vs After Augmentation**

## 🔬 Future Work

- Incorporating attention modules for channel/spatial awareness.
- Adding full training loop and evaluation metrics.
- Exporting trained model for on-device usage.

## 👩‍💻 Author

**Tanish Dalal**  
- Research Assistant @ Queen Mary University of London  
- AI Intern @ IBM  
- AI Trainer @ Outlier  
- Aspiring AI-ML Engineer  

---

Let me know if you’d like a badge section, training logs, loss/accuracy plots, or how to run this as a full training pipeline too!
