# 📝 Handwritten Text Recognition using DenseNet and Attention-based RNN

This project implements a deep learning model for offline handwritten text recognition. The architecture combines a CNN-based **DenseNet encoder** with an **Attention-based RNN decoder**, allowing the model to extract meaningful features from handwritten word images and generate the corresponding word sequence.

---

## 📚 Overview

- **Encoder:** Custom DenseNet121 for feature extraction from image inputs.
- **Decoder:** Attention-based GRU decoder to convert visual features into a character sequence.
- **Training Strategy:** Teacher forcing with NLLLoss and evaluation using Word Error Rate (WER).
- **Goal:** Automatically recognize handwritten words from scanned image data.

---

## 🏁 Project Structure

| File | Description |
|------|-------------|
| `Train.py` | Main training script. Loads data, trains the model, evaluates WER/SACC, and saves checkpoints. |
| `Attention_RNN.py` | Implements the GRU-based decoder with attention mechanism. |
| `Densenet_torchvision.py` | Custom implementation of DenseNet-121 CNN for use as an encoder. |
| `data_iterator.py` | Loads and batches data from `.pkl` and label files. |
| `Densenet_testway.py` | Evaluation script to load saved models and calculate WER on test data. |
| `gen_pkl.py` | One-time preprocessing script to convert `.bmp` images into a `.pkl` format for training. |

---

## 📦 Data Format

- **Images:** Single-channel `.bmp` format.
- **Labels:** Plain text file (`.txt`) in format: `image_id word1 word2 ...`
- **Dictionary:** Maps each word/token to a unique integer ID (`dictionary.txt`).
- **Feature File:** Pickle file (`offline-train.pkl`) storing image tensors of shape `[1, H, W]`.

---

## 🧪 Training

Before training, make sure your data has been converted using `gen_pkl.py`.
## BEST RESULT
| Metric                       | Value     |
| ---------------------------- | --------- |
| **WER** (Word Error Rate)    | `0.18241` |
| **SACC** (Sequence Accuracy) | `0.36324` |

