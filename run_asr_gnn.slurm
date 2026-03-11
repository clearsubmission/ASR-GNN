#!/bin/bash
#SBATCH --account=def-kobti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=asr_gnn
#SBATCH --output=logs/asr_gnn-%A_%a.out
#SBATCH --error=logs/asr_gnn-%A_%a.err

set -euo pipefail

module purge
module load python/3.11 cuda/12.2
source ~/torch-env/bin/activate

cd ~/kg_experiments
mkdir -p logs
mkdir -p /scratch/amangel/asr_runs

DATASETS=("FB15k237" "WN18RR" "YAGO310")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

OUTDIR="/scratch/amangel/asr_runs/${DATASET}_seed42"

python scripts/train_asr_gnn.py \
  --dataset "$DATASET" \
  --output_dir "$OUTDIR" \
  --emb_dim 200 \
  --num_layers 2 \
  --hidden_dim 256 \
  --dropout 0.2 \
  --lambda_l0 1e-4 \
  --lr 1e-3 \
  --batch_size 1024 \
  --epochs 100 \
  --num_negatives 10 \
  --eval_batch_size 128 \
  --seed 42
