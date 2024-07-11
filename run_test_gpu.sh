#!/bin/bash
#SBATCH --job-name=mlm_bert_stats     # nom du job
#SBATCH -C a100
#SBATCH -A mwd@a100
#SBATCH --ntasks=96                 # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --gres=gpu:8                 # nombre de GPU par nœud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
#SBATCH --cpus-per-task=10           # nombre de CPU par tache (un quart du noeud ici)
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=./logs/mlm_test%j.out # nom du fichier de sortie
#SBATCH --error=./logs/mlm_test%j.out  # nom du fichier d'erreur (ici commun avec la sortie)


# Envoi des mails
#SBATCH --mail-type=begin,fail,abort,end
 
# Nettoyage des modules charges en interactif et herites par defaut
module purge

# Chargement des modules
module load pytorch-gpu/py3/2.3.0
#
#python3.9 -m pip install --user --no-cache-dir -r requirements_training.txt
 
# Echo des commandes lancees
set -x -e

export OMP_NUM_THREADS=10

export CUDA_LAUNCH_BLOCKING=1

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1

srun -l python -u run_test_gpu.py
    

