#!/bin/bash
# Setup del entorno en Hercules
# Ejecutar UNA VEZ antes del primer sbatch: bash hercules/setup_env.sh

set -e

echo "=== Configurando entorno QMAML en Hercules ==="

# Módulos
module purge
module load python/3.10
module load cuda/12.1
module load miniconda/23.5

# Crear entorno conda
conda create -n qmaml-env python=3.10 -y
source activate qmaml-env

# Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# Verificar
python -c "import pennylane, torch, higher; print('OK — PennyLane', pennylane.__version__, '| PyTorch', torch.__version__)"

# Pre-descargar Omniglot
python -c "
import torchvision
torchvision.datasets.Omniglot(root='data/', background=True, download=True)
torchvision.datasets.Omniglot(root='data/', background=False, download=True)
print('Omniglot descargado')
"

# Mini test
echo "=== Mini test ==="
python code/mini_test.py

echo "=== Setup completo — listo para sbatch ==="
echo "Lanzar con: sbatch hercules/submit_array.slurm"
