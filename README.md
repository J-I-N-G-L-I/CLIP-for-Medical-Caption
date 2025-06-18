# CLIP-for-Medical-Caption

## Project Overview

This repository provides a practice implementation of OpenAI’s CLIP (Contrastive Language–Image Pretraining) model, fine-tuned for medical image captioning. We jointly train the image and text encoders so the model can generate or match diagnostic descriptions (captions) for chest CT/X-ray images.

## Dataset

The project uses the public ROCOv2 dataset (Radiology Objects in COntext v2) from Kaggle:

* **ROCO v2 Dataset**: [https://www.kaggle.com/datasets/claudiopisa9884/roco-v2](https://www.kaggle.com/datasets/claudiopisa9884/roco-v2)

  * Contains images (.jpg) and corresponding captions (.csv) for train/validation/test splits.

## Requirements

* Python ≥3.8
* PyTorch
* torchvision
* transformers (Hugging Face)
* pandas
* Pillow
* tqdm
* tensorboard

Install dependencies via:

```bash
pip install -r requirements.txt
```

## Directory Structure

```
CLIP-for-Medical-Caption/
├── config/                   
│   ├── __init__.py
│   └── config.py             # Model & training hyperparameters
├── evaluation/               
│   ├── __init__.py
│   ├── evaluator.py          # Retrieval metrics evaluator
│   └── metrics.py            # Recall@K, Median Rank, etc.
├── scripts/                  
│   ├── train.py              # Training script
│   ├── inference.py          # Inference script
│   └── evaluate.py           # Evaluation script
├── src/                      
│   ├── dataset/              # Dataset and transforms
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── models/               # Model definitions
│   │   ├── __init__.py
│   │   ├── clip_model.py
│   │   ├── vision_encoder.py
│   │   ├── text_encoder.py
│   │   └── losses.py
│   └── training/             # Training & evaluation logic
│       ├── __init__.py
│       └── trainer.py        # Trainer class with optimizer, scheduler, mixed-precision
├── checkpoints/              # Saved model checkpoints
├── logs/                     # TensorBoard logs
├── README.md                 # This README file
├── requirements.txt          # Python dependencies
└── train.slurm               # SLURM job submission script (For HPC)
```

## Getting Started

### 1. Data Preparation

1. Download the ROCOv2 dataset from Kaggle and extract files.
2. Edit `TrainingConfig.data_root` in `config/config.py` to point to your local `data_root`.

### 2. Training

Run the training script:

```bash
python scripts/train.py \
  --config config/config.py \
  --data_root /path/to/rocov2 \
  --batch_size 128 \
  --max_epochs 30
```

* Checkpoints and the best model are saved under `checkpoints/`.
* TensorBoard logs are stored in `logs/`:

  ```bash
  tensorboard --logdir logs/
  ```

### 3. Inference

To rank candidate captions for a given image, use:

```bash
python scripts/inference.py \
  --checkpoint checkpoints/best_model.pth \
  --image /path/to/sample.jpg \
  --texts "Normal chest X-ray" "Cardiomegaly" "No acute findings"
```
or one line
```bash
python scripts/inference.py --checkpoint checkpoints/best_model.pth --image /scratch/sc232jl/rocov2/test_images/test/ROCOv2_2023_test_000001.jpg --texts "CT chest axial view showing a huge ascending aortic aneurysm (*)." "normal chest" "cardiomegaly present"
```

The script outputs similarity scores (higher means more relevant) and sorts the candidates.

### 4. Evaluation

Compute retrieval metrics on a split (e.g., validation):

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/best_model.pth \
  --data_root /path/to/rocov2 \
  --split valid
```

Supported metrics include Recall\@1, Recall\@5, Recall\@10, and Median Rank.

## Hyperparameters & Configuration

* **Image Encoder**: ResNet50 or ViT
* **Text Max Length**: 77 tokens (CLIP native)
* **Embedding Dimension**: 512
* **Learning Rate**: 5e-4 with linear warm-up + cosine decay
* **Batch Size**: 128
* **Optimizer**: AdamW
* **Loss**: Contrastive cross-entropy with learnable temperature

Adjust these settings in `config/config.py` as needed.

*This project is for research and learning only. Contributions via Issues and Pull Requests are welcome!*
