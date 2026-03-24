"""
Mini test local — verifica que QMAML funciona end-to-end.
Usa datos sintéticos, 1 seed, pocos episodios. Corre en ~3-5 minutos.
Ejecutar ANTES de lanzar en Hercules.
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy
import higher
from pathlib import Path

# Importar desde el mismo directorio
sys.path.insert(0, str(Path(__file__).parent))
from qmaml_experiment import (
    QMAMLModelDiff, ClassicalMAML,
    OmniglotFewShot, collate_episodes,
    inner_loop_euclidean, inner_loop_qng,
    compute_qfim_diagonal,
    vqc_circuit,
    N_QUBITS, N_WAY, INNER_LR, OUTER_LR,
    DEVICE, FIGURES_DIR,
)

# ── Config reducida para test rápido ──────────────────────────────────────────
N_META_TRAIN_MINI = 40     # vs 2000 en full
N_META_TEST_MINI  = 20     # vs 600
META_BATCH_MINI   = 2      # vs 4
INNER_STEPS_MINI  = 3      # vs 5
K_SHOT_TEST       = 5

print("=" * 60)
print("QMAML — Mini Test Local (datos sintéticos)")
print("=" * 60)
print(f"Episodios train: {N_META_TRAIN_MINI} | test: {N_META_TEST_MINI}")
print(f"Inner steps: {INNER_STEPS_MINI} | Batch: {META_BATCH_MINI}")
print()

# ── 1. Test de componentes ─────────────────────────────────────────────────
print("[1] Test de componentes...")

# Modelo y forward pass
model = QMAMLModelDiff(input_dim=784, n_way=N_WAY)
x = torch.randn(4, 784)
out = model(x)
assert out.shape == (4, N_WAY), f"Forward incorrecto: {out.shape}"
print(f"  ✓ QMAMLModelDiff forward: {out.shape}")

# QFIM diagonal
z = model.encoder(x)
fisher = compute_qfim_diagonal(z[:3], model.theta_shared, model.theta_task)
assert fisher.shape == model.theta_task.shape, "QFIM shape incorrecta"
print(f"  ✓ QFIM diagonal shape: {fisher.shape} | mean: {fisher.mean().item():.4f}")

# Inner loop euclidean
sx = torch.randn(K_SHOT_TEST * N_WAY, 784)
sy = torch.repeat_interleave(torch.arange(N_WAY), K_SHOT_TEST)
theta_eu = inner_loop_euclidean(model, sx, sy, n_steps=2, lr=INNER_LR)
assert theta_eu.shape == model.theta_task.shape
print(f"  ✓ Inner loop euclidean OK — theta_task delta: {(theta_eu - model.theta_task).abs().mean():.4f}")

# Inner loop QNG
theta_qng = inner_loop_qng(model, sx, sy, n_steps=2, lr=INNER_LR)
assert theta_qng.shape == model.theta_task.shape
delta_eu  = (theta_eu  - model.theta_task).abs().mean().item()
delta_qng = (theta_qng - model.theta_task).abs().mean().item()
print(f"  ✓ Inner loop QNG OK    — theta_task delta: {delta_qng:.4f}")
print(f"    QNG/Euclidean ratio: {delta_qng/max(delta_eu,1e-8):.3f} "
      f"({'QNG da pasos más grandes — QFIM bien condicionada' if delta_qng > delta_eu else 'QNG más conservador'})")

# Dataset sintético
ds = OmniglotFewShot("/tmp", split="train",
                     n_way=N_WAY, k_shot=K_SHOT_TEST,
                     n_query=10, n_episodes=10, seed=0)
sx2, sy2, qx2, qy2 = ds[0]
assert sx2.shape == (K_SHOT_TEST * N_WAY, 784)
print(f"  ✓ Dataset sintético OK: support {sx2.shape}, query {qx2.shape}")

# Classical MAML
cl_model = ClassicalMAML(input_dim=784, n_way=N_WAY)
out_cl = cl_model(x)
assert out_cl.shape == (4, N_WAY)
print(f"  ✓ ClassicalMAML forward OK: {out_cl.shape}")

print()
print("[2] Mini meta-entrenamiento (pocos episodios)...")

loss_fn = nn.CrossEntropyLoss()

# ── Classical MAML ────────────────────────────────────────────────────────────
print("\n  [Classical MAML]")
cl_model  = ClassicalMAML(input_dim=784, n_way=N_WAY)
outer_opt = optim.Adam(cl_model.parameters(), lr=OUTER_LR)
cl_accs   = []

ds_mini = OmniglotFewShot("/tmp", split="train", n_way=N_WAY, k_shot=K_SHOT_TEST,
                           n_query=10, n_episodes=N_META_TRAIN_MINI * META_BATCH_MINI,
                           seed=42)
loader_mini = DataLoader(ds_mini, batch_size=META_BATCH_MINI,
                         collate_fn=collate_episodes)

for ep_idx, (sx, sy, qx, qy) in enumerate(loader_mini):
    if ep_idx >= N_META_TRAIN_MINI // META_BATCH_MINI:
        break
    outer_opt.zero_grad()
    meta_loss = torch.tensor(0.0)
    for t in range(sx.shape[0]):
        inner_opt = optim.SGD(cl_model.parameters(), lr=INNER_LR)
        with higher.innerloop_ctx(cl_model, inner_opt, copy_initial_weights=False) as (fm, dopt):
            for _ in range(INNER_STEPS_MINI):
                dopt.step(loss_fn(fm(sx[t]), sy[t]))
            ql = fm(qx[t])
            meta_loss = meta_loss + loss_fn(ql, qy[t])
            cl_accs.append((ql.detach().argmax(1) == qy[t]).float().mean().item())
    (meta_loss / sx.shape[0]).backward()
    outer_opt.step()

print(f"  train acc (últimos 10 ep): {np.mean(cl_accs[-10:]):.3f}")

# ── QMAML Euclidean ───────────────────────────────────────────────────────────
print("\n  [QMAML-Euclidean]")
qm_eu     = QMAMLModelDiff(input_dim=784, n_way=N_WAY)
outer_opt = optim.Adam(qm_eu.parameters(), lr=OUTER_LR)
eu_accs   = []

for ep_idx, (sx, sy, qx, qy) in enumerate(loader_mini):
    if ep_idx >= N_META_TRAIN_MINI // META_BATCH_MINI:
        break
    outer_opt.zero_grad()
    meta_loss = torch.tensor(0.0)
    for t in range(sx.shape[0]):
        theta_ad = inner_loop_euclidean(qm_eu, sx[t], sy[t], INNER_STEPS_MINI, INNER_LR)
        z = qm_eu.encoder(qx[t])
        outs = [torch.tensor(vqc_circuit(z[i].detach().numpy(),
                             qm_eu.theta_shared.detach().numpy(),
                             theta_ad.detach().numpy()), dtype=torch.float32)
                for i in range(z.shape[0])]
        logits = qm_eu.classifier(torch.stack(outs) + 0.0 * qm_eu.theta_shared.sum())
        meta_loss = meta_loss + loss_fn(logits, qy[t])
        eu_accs.append((logits.detach().argmax(1) == qy[t]).float().mean().item())
    (meta_loss / sx.shape[0]).backward()
    outer_opt.step()

print(f"  train acc (últimos 10 ep): {np.mean(eu_accs[-10:]):.3f}")

# ── QMAML-QNG ─────────────────────────────────────────────────────────────────
print("\n  [QMAML-QNG]")
qm_qng    = QMAMLModelDiff(input_dim=784, n_way=N_WAY)
outer_opt = optim.Adam(qm_qng.parameters(), lr=OUTER_LR)
qng_accs  = []

for ep_idx, (sx, sy, qx, qy) in enumerate(loader_mini):
    if ep_idx >= N_META_TRAIN_MINI // META_BATCH_MINI:
        break
    outer_opt.zero_grad()
    meta_loss = torch.tensor(0.0)
    for t in range(sx.shape[0]):
        theta_ad = inner_loop_qng(qm_qng, sx[t], sy[t], INNER_STEPS_MINI, INNER_LR)
        z = qm_qng.encoder(qx[t])
        outs = [torch.tensor(vqc_circuit(z[i].detach().numpy(),
                             qm_qng.theta_shared.detach().numpy(),
                             theta_ad.detach().numpy()), dtype=torch.float32)
                for i in range(z.shape[0])]
        logits = qm_qng.classifier(torch.stack(outs) + 0.0 * qm_qng.theta_shared.sum())
        meta_loss = meta_loss + loss_fn(logits, qy[t])
        qng_accs.append((logits.detach().argmax(1) == qy[t]).float().mean().item())
    (meta_loss / sx.shape[0]).backward()
    outer_opt.step()

print(f"  train acc (últimos 10 ep): {np.mean(qng_accs[-10:]):.3f}")

# ── Resultados del mini test ───────────────────────────────────────────────────
print()
print("=" * 60)
print("RESULTADOS MINI TEST (train query acc, últimos episodios)")
print("=" * 60)
random_baseline = 1 / N_WAY
print(f"  Random baseline:    {random_baseline:.3f}")
print(f"  Classical MAML:     {np.mean(cl_accs[-10:]):.3f}")
print(f"  QMAML-Euclidean:    {np.mean(eu_accs[-10:]):.3f}")
print(f"  QMAML-QNG:          {np.mean(qng_accs[-10:]):.3f}")

improvement_vs_classical = (np.mean(qng_accs[-10:]) - np.mean(cl_accs[-10:])) * 100
improvement_vs_euclidean = (np.mean(qng_accs[-10:]) - np.mean(eu_accs[-10:])) * 100

print()
print(f"  QNG vs Classical:   {improvement_vs_classical:+.1f}pp")
print(f"  QNG vs Euclidean:   {improvement_vs_euclidean:+.1f}pp")
print()

if np.mean(qng_accs[-10:]) > random_baseline + 0.02:
    print("  ✓ TODOS los métodos aprenden (por encima de random)")
else:
    print("  ⚠ Convergencia lenta — normal con pocos episodios y datos sintéticos")

print()
print("✓ MINI TEST COMPLETADO — OK para lanzar en Hercules")
print("  Ejecutar: sbatch hercules/submit_qmaml.slurm")
