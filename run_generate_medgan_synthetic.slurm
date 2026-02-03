#!/bin/bash
#SBATCH --account=jalenj4-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jalen.jiang2@gmail.com

cd /u/jalenj4/PyHealth-Medgan-Corgan-Port
source /u/jalenj4/pehr_scratch/venv/bin/activate
export PYTHONPATH=/u/jalenj4/PyHealth-Medgan-Corgan-Port:$PYTHONPATH

echo "Generating 10k synthetic patients using MedGAN - COUNT MODE"
echo "Checkpoint: /scratch/jalenj4/medgan_45k_count_results/medgan_final.pth"
echo "Output: /scratch/jalenj4/synthetic_medgan_count_10k.csv"

python examples/generate_synthetic_mimic3_medgan.py \
    --checkpoint /scratch/jalenj4/medgan_45k_count_results/medgan_final.pth \
    --vocab /scratch/jalenj4/medgan_45k_count_results/icd9_vocabulary.txt \
    --data_matrix /scratch/jalenj4/medgan_45k_count_results/icd9_count_matrix.npy \
    --output /scratch/jalenj4/synthetic_medgan_count_10k.csv \
    --n_samples 10000 \
    --data_mode count \
    --count_activation relu \
    --count_loss mse
