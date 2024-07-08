#!/bin/bash
#SBATCH --job-name=jeanzay
#SBATCH --nodes=3
#SBATCH --ntasks=24            # Nombre total de processus MPI
#SBATCH --ntasks-per-node=8    # Nombre de processus MPI par noeud
#SBATCH --hint=nomultithread   # 1 processus MPI par coeur physique (pas d'hyperthreading)
##SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=19:00:00
#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --error=./logs/%x_%A_%a.err
#SBATCH -A mwd@v100

module purge
module load pytorch-gpu/py3/1.11.0

nvidia-smi

srun python preprocessing_dataset.py \
    --do_train \
    --overwrite_output_dir \
    --max_seq_length=512 \
    --log_level='info' \
    --logging_first_step='True' \
    --cache_dir='/cache/' \
    --path_save_dataset="/data/tokenized_dataset" \
    --output_dir='/output' \
    --preprocessing_num_workers=20

