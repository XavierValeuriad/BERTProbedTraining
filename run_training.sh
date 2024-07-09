#!/bin/bash
#SBATCH --job-name=mlm_bert_stats     # nom du job
#SBATCH --constraint=v100-32g
#SBATCH --ntasks=4                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=4          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:4                 # nombre de GPU par nœud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
#SBATCH --cpus-per-task=10           # nombre de CPU par tache (un quart du noeud ici)
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=0:20:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=./logs/mlm_test%j.out # nom du fichier de sortie
#SBATCH --error=./logs/mlm_test%j.out  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH -A mwd@v100

# Envoi des mails
#SBATCH --mail-type=begin,fail,abort,end
 
# Nettoyage des modules charges en interactif et herites par defaut
module purge

# Chargement des modules
module load pytorch-gpu/py3/1.12.1
 
# Echo des commandes lancees
set -x -e

export OMP_NUM_THREADS=10

export CUDA_LAUNCH_BLOCKING=1

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1

srun -l python -u run_training.py \
    --num_train_epochs=1 \
    --save_steps=2 \
    --logging_steps=300 \
    --model_type='bert-base-uncased' \
    --config_overrides="max_position_embeddings=514,type_vocab_size=1,vocab_size=32005,bos_token_id=5,eos_token_id=6" \
    --path_load_dataset="data/tokenized_train_bert_1" \
    --output_dir='model_output/' \
    --logging_dir='model_output/logs/' \
    --per_device_train_batch_size=32 \
    --do_train \
    --warmup_steps=10000 \
    --overwrite_output_dir \
    --max_seq_length=512 \
    --report_to='tensorboard' \
    --save_strategy='steps' \
    --skip_memory_metrics='False' \
    --log_level='info' \
    --seed=42 \
    --data_seed=42 \
    --logging_first_step='True' \
    --fp16 \
    --ddp_timeout=600 \
    --ddp_find_unused_parameters='False' \
    

