#!/bin/bash
#SBATCH --job-name=qmaml_full
#SBATCH --output=logs/qmaml_%j.out
#SBATCH --error=logs/qmaml_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=danitanke02@gmail.com

# ── Hercules (Junta de Andalucía) — QMAML Full Experiment ────────────────────
# Supercomputador: https://www.juntadeandalucia.es/servicios/sede/tramites/hecules
# Módulos disponibles: Python 3.10+, CUDA 12, miniconda

echo "============================================"
echo "QMAML Full Experiment — Hercules"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodo: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Inicio: $(date)"
echo "============================================"

# ── Módulos ───────────────────────────────────────────────────────────────────
module purge
module load python/3.10
module load cuda/12.1           # si hay GPU disponible
module load miniconda/23.5

# ── Entorno conda ──────────────────────────────────────────────────────────────
source activate qmaml-env 2>/dev/null || {
    conda create -n qmaml-env python=3.10 -y
    conda activate qmaml-env
    pip install -r ../requirements.txt
}

# ── Directorios ────────────────────────────────────────────────────────────────
mkdir -p logs results figures data

# ── Verificar mini test antes de lanzar full ──────────────────────────────────
echo "[Pre-check] Ejecutando mini test..."
python code/mini_test.py
if [ $? -ne 0 ]; then
    echo "ERROR: mini test falló. Abortando."
    exit 1
fi
echo "[Pre-check] OK — lanzando experimento completo"

# ── Experimento completo ──────────────────────────────────────────────────────
python code/qmaml_experiment_hpc.py \
    --n-qubits 8 \
    --n-shared 3 \
    --n-task 2 \
    --n-meta-train 10000 \
    --n-meta-test 2000 \
    --n-seeds 5 \
    --n-workers $SLURM_CPUS_PER_TASK \
    --output-dir results/

echo "============================================"
echo "Fin: $(date)"
echo "============================================"
