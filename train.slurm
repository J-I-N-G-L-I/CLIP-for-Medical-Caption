#!/bin/bash
#SBATCH --job-name=training_clip_medical_caption
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=30:00:00
#SBATCH --output=train_%j.out

# load environment
module load miniforge
source activate clip_medical_caption

cd /users/sc232jl/CLIP_for_Medical_Caption

python ./scripts/train.py \
  --data_root /scratch/sc232jl/rocov2 \
  --save_dir ./checkpoints \
  --batch_size 128 \
  --epochs 30 \
  --lr 5e-4
