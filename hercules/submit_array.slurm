#!/bin/bash
#SBATCH --job-name=qmaml_seed
#SBATCH --output=logs/qmaml_%A_%a.out
#SBATCH --error=logs/qmaml_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-4          # 5 seeds en paralelo
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=danitanke02@gmail.com

# ── Job Array: 1 seed por nodo, todos en paralelo ─────────────────────────────
# Lanzar con: sbatch submit_array.slurm
# Monitorear: squeue -u $USER
# Ver logs:   tail -f logs/qmaml_<ARRAY_ID>_<SEED>.out

SEED=$SLURM_ARRAY_TASK_ID

echo "============================================"
echo "QMAML — Seed $SEED / Job Array $SLURM_ARRAY_JOB_ID"
echo "Nodo: $SLURM_NODELIST | CPUs: $SLURM_CPUS_PER_TASK"
echo "Inicio: $(date)"
echo "============================================"

module purge
module load python/3.10
module load cuda/12.1
module load miniconda/23.5
source activate qmaml-env

mkdir -p logs results/$SEED figures data

python code/qmaml_experiment_hpc.py \
    --seed $SEED \
    --n-qubits 8 \
    --n-shared 3 \
    --n-task 2 \
    --n-meta-train 10000 \
    --n-meta-test 2000 \
    --n-workers $SLURM_CPUS_PER_TASK \
    --output-dir results/$SEED/

echo "Seed $SEED completado: $(date)"
