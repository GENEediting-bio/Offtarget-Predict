# Nucleotide Transformer Fine-tuning and Prediction

A PyTorch implementation for fine-tuning the Nucleotide Transformer model on biological sequence classification tasks, with local prediction capabilities to avoid network dependencies.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Download](#model-download)
- [Training](#training)
- [Prediction](#prediction)
- [Input Format](#input-format)
- [Output Format](#output-format)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project provides tools to fine-tune the Nucleotide Transformer model (by InstaDeepAI) for DNA/RNA sequence classification tasks. The trained model can predict whether sequences belong to positive or negative classes with high accuracy.

**Pre-trained Model Checkpoint**: 
- `best_epoch20_auc0.9526.pt` (AUC: 0.9526) - Available on [Google Drive](https://drive.google.com/file/d/1t0lx1lYniOk8wYLxI5Igm4AzsfPuwqDX/view?usp=drive_link)

## ✨ Features

- 🧬 Fine-tune Nucleotide Transformer on custom sequence datasets
- 🔄 Local model loading (no internet required for prediction)
- 📊 Comprehensive evaluation metrics (AUC, Accuracy, F1-score)
- ⚡ Batch prediction for large datasets
- 🎯 High-accuracy classification (AUC > 0.95)
- 🔧 Flexible training parameters

## 🚀 Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA-capable GPU (recommended)

### Install Dependencies

```bash
pip install torch transformers pandas numpy scikit-learn tqdm
```

## 📥 Model Download

### Pre-trained Checkpoint

Download the fine-tuned model checkpoint:

```bash
# Download from Google Drive
# Place the file in your project directory:
# best_epoch20_auc0.9526.pt
```

### Base Model Setup

The prediction script can automatically download the base Nucleotide Transformer model:

```bash
# Download base model to local directory
python finetune_model_predict.py --download_model
```

This will create a `local_models/` directory containing the model files for offline use.

## 🏃 Quick Start

### 1. Prepare Your Data

Create a CSV file with your sequences:

```csv
sequence
ATCGATCGATCG
GCTAGCTAGCTA
TTTTAAAACCCC
```

### 2. Run Prediction

```bash
python finetune_model_predict.py \
    --checkpoint best_epoch20_auc0.9526.pt \
    --input_csv your_sequences.csv \
    --output_csv predictions.csv \
    --download_model  # Auto-download base model if needed
```

## 🎯 Prediction

### Basic Usage

```bash
python finetune_model_predict.py \
    --checkpoint best_epoch20_auc0.9526.pt \
    --input_csv example_input.csv \
    --output_csv prediction_output.csv
```

### Advanced Options

```bash
python finetune_model_predict.py \
    --checkpoint best_epoch20_auc0.9526.pt \
    --input_csv example_input.csv \
    --output_csv prediction_output.csv \
    --local_model_dir ./local_models/nucleotide-transformer-500m-1000g \
    --batch_size 32 \
    --max_length 512 \
    --device cuda
```


### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | **Required** | Path to trained model checkpoint (.pt file) |
| `--input_csv` | **Required** | Input CSV file with sequences |
| `--output_csv` | **Required** | Output CSV file for predictions |
| `--local_model_dir` | `./local_models/...` | Local directory containing base model |
| `--batch_size` | `16` | Batch size for prediction |
| `--max_length` | `512` | Maximum sequence length |
| `--device` | `cuda` | Device for inference (`cuda` or `cpu`) |
| `--download_model` | `False` | Auto-download base model if missing |

## 📊 Training

### Training Script

Use the fine-tuning script to train on your own data:

```bash
python finetune_nt_pytorch_fixed.py \
    --train_csv train_data.csv \
    --dev_csv dev_data.csv \
    --test_csv test_data.csv \
    --epochs 20 \
    --batch_size 16 \
    --lr 2e-5 \
    --ckpt_dir ./checkpoints
```

### Training Parameters

- **Learning Rate**: 2e-5 (recommended)
- **Batch Size**: 8-32 (depending on GPU memory)
- **Epochs**: 10-20
- **Sequence Length**: Up to 1000 tokens

## 📁 Input Format

### Required CSV Structure

The input CSV should contain at least one column with DNA/RNA sequences:

```csv
sequence
ATCGATCGATCGATCG
GCTAGCTAGCTAGCTA
TTTTAAAACCCCGGGG
```

### Supported Column Names

The script automatically detects sequence columns with these names:
- `sequence`
- `seq` 
- `dna`
- `rna`
- Or the first column if no matches found

## 📄 Output Format

The prediction output includes:

```csv
sequence,prediction,probability_class_0,probability_class_1,confidence,predicted_label
ATCGATCGATCG,1,0.023,0.977,0.977,positive
GCTAGCTAGCTA,0,0.891,0.109,0.891,negative
```

### Output Columns

| Column | Description |
|--------|-------------|
| `prediction` | Binary prediction (0 or 1) |
| `probability_class_0` | Probability of negative class |
| `probability_class_1` | Probability of positive class |
| `confidence` | Maximum prediction confidence |
| `predicted_label` | Human-readable label |

## 🔧 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```bash
   # Ensure the checkpoint file exists
   ls -la best_epoch20_auc0.9526.pt
   ```

2. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   --batch_size 8
   # Or use CPU
   --device cpu
   ```

3. **Missing Base Model**
   ```bash
   # Auto-download
   --download_model
   # Or specify local path
   --local_model_dir /path/to/local/model
   ```

### Performance Tips

- Use `--device cuda` for GPU acceleration
- Adjust `--batch_size` based on available GPU memory
- For long sequences, increase `--max_length` (up to 1000)
- Use `--download_model` once, then reuse local model

## 📈 Results

The provided checkpoint achieves:
- **AUC**: 0.9526
- **Accuracy**: >90%
- **F1-Score**: >0.90

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is for academic and research use. Please check the original [Nucleotide Transformer](https://github.com/instadeepai/nucleotide-transformer) license for commercial use.  

## 🙏 Acknowledgments

- [InstaDeepAI](https://www.instadeep.com/) for the Nucleotide Transformer model
- Hugging Face for the Transformers library
- The bioinformatics community for datasets and tools

---

**Note**: The `best_epoch20_auc0.9526.pt` checkpoint file is available for download via Google Drive. Please contact the maintainers for access.
