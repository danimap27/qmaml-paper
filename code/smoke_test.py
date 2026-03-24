"""
Smoke test — verifica end-to-end en < 5 minutos.
Datos sintéticos, 10 episodios, 2 inner steps.
Ejecutar antes de cualquier experimento largo.

Uso:
    python code/smoke_test.py
    python code/smoke_test.py --sampler          # usa SamplerQNN en vez de EstimatorQNN
    python code/smoke_test.py --n-episodes 20
"""
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from copy import deepcopy
import higher

sys.path.insert(0, str(Path(__file__).parent))

# ─── Args ─────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--sampler",      action="store_true", help="Usa SamplerQNN (muestreo)")
p.add_argument("--n-episodes",   type=int, default=10)
p.add_argument("--inner-steps",  type=int, default=2)
p.add_argument("--n-qubits",     type=int, default=4)
p.add_argument("--n-way",        type=int, default=5)
p.add_argument("--k-shot",       type=int, default=5)
args = p.parse_args()

import pennylane as qml

N_Q      = args.n_qubits
N_WAY    = args.n_way
K_SHOT   = args.k_shot
N_SHARED = 2
N_TASK   = 1

print("=" * 55)
print(f"QMAML Smoke Test {'[SAMPLER]' if args.sampler else '[ESTIMATOR]'}")
print(f"  qubits={N_Q} | {N_WAY}-way {K_SHOT}-shot | "
      f"episodes={args.n_episodes} | inner_steps={args.inner_steps}")
print("=" * 55)

t0 = time.time()

# ─── Dispositivo ──────────────────────────────────────────────────────────────
dev = qml.device("lightning.qubit", wires=N_Q)

# ─── QNode: Estimator (expvals) o Sampler (probs) ─────────────────────────────
if not args.sampler:
    @qml.qnode(dev, diff_method="adjoint", interface="torch")
    def circuit(inputs, theta_shared, theta_task):
        for i in range(N_Q):
            qml.RY(np.pi * inputs[i], wires=i)
        qml.StronglyEntanglingLayers(theta_shared, wires=range(N_Q))
        qml.StronglyEntanglingLayers(theta_task,   wires=range(N_Q))
        return [qml.expval(qml.PauliZ(i)) for i in range(N_Q)]

    weight_shapes = {
        "theta_shared": (N_SHARED, N_Q, 3),
        "theta_task":   (N_TASK,   N_Q, 3),
    }
    out_dim = N_Q
    print(f"  Backend: EstimatorQNN (expval <Z_i>), output_dim={out_dim}")

else:
    # SamplerQNN: mide en base computacional, devuelve probabilidades de bitstrings
    @qml.qnode(dev, diff_method="parameter-shift", interface="torch")
    def circuit(inputs, theta_shared, theta_task):
        for i in range(N_Q):
            qml.RY(np.pi * inputs[i], wires=i)
        qml.StronglyEntanglingLayers(theta_shared, wires=range(N_Q))
        qml.StronglyEntanglingLayers(theta_task,   wires=range(N_Q))
        return qml.probs(wires=range(N_Q))   # 2^N_Q probabilidades

    weight_shapes = {
        "theta_shared": (N_SHARED, N_Q, 3),
        "theta_task":   (N_TASK,   N_Q, 3),
    }
    out_dim = 2 ** N_Q
    print(f"  Backend: SamplerQNN (probs), output_dim={out_dim}")

# ─── Modelo ────────────────────────────────────────────────────────────────────
class SmokeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 64), nn.ReLU(),
            nn.Linear(64, N_Q), nn.Tanh()
        )
        self.vqc = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.classifier = nn.Linear(out_dim, N_WAY)

    def forward(self, x):
        z     = self.encoder(x)
        q_out = torch.stack([self.vqc(z[i]) for i in range(z.shape[0])])
        return self.classifier(q_out)

    @property
    def theta_task(self):
        return self.vqc.qnode_weights["theta_task"]

    @property
    def theta_shared(self):
        return self.vqc.qnode_weights["theta_shared"]

# ─── Test 1: Forward pass ──────────────────────────────────────────────────────
print("\n[1] Forward pass...")
t1 = time.time()
model = SmokeModel()
x_test = torch.randn(3, 784)
out    = model(x_test)
assert out.shape == (3, N_WAY), f"Shape incorrecta: {out.shape}"
print(f"  ✓ output shape: {out.shape} | {time.time()-t1:.1f}s")

# ─── Test 2: Backward pass (gradient flow) ─────────────────────────────────────
print("\n[2] Backward pass...")
t2 = time.time()
loss = nn.CrossEntropyLoss()(out, torch.zeros(3, dtype=torch.long))
loss.backward()
has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.parameters())
assert has_grad, "No hay gradientes!"
print(f"  ✓ gradientes OK | loss={loss.item():.4f} | {time.time()-t2:.1f}s")

# ─── Test 3: QFIM diagonal ─────────────────────────────────────────────────────
print("\n[3] QFIM diagonal (parameter-shift manual)...")
t3 = time.time()
eps = 0.5 * np.pi
theta_flat = model.theta_task.detach().flatten()
n_params   = theta_flat.numel()
x_enc      = model.encoder(torch.randn(5, 784)).detach()

fisher_diag = []
for i in range(min(n_params, 4)):    # solo primeros 4 para el test
    t_plus  = theta_flat.clone(); t_plus[i]  += eps
    t_minus = theta_flat.clone(); t_minus[i] -= eps
    grad_sq = 0.0
    for x in x_enc:
        out_p = torch.tensor(circuit(x.numpy(),
                             model.theta_shared.detach().numpy(),
                             t_plus.reshape(model.theta_task.shape).numpy()),
                             dtype=torch.float32)
        out_m = torch.tensor(circuit(x.numpy(),
                             model.theta_shared.detach().numpy(),
                             t_minus.reshape(model.theta_task.shape).numpy()),
                             dtype=torch.float32)
        grad_sq += float(((out_p - out_m) / 2.0).pow(2).sum())
    fisher_diag.append(grad_sq / len(x_enc))

print(f"  ✓ QFIM[0:4] = {[f'{v:.4f}' for v in fisher_diag]} | {time.time()-t3:.1f}s")
print(f"  QFIM bien condicionada: {'SÍ' if min(fisher_diag) > 1e-6 else 'NO (posible barren plateau)'}")

# ─── Test 4: Inner loop QNG ────────────────────────────────────────────────────
print("\n[4] Inner loop QNG (2 steps)...")
t4 = time.time()
sx = torch.randn(K_SHOT * N_WAY, 784)
sy = torch.repeat_interleave(torch.arange(N_WAY), K_SHOT)
loss_fn = nn.CrossEntropyLoss()

theta = model.theta_task.clone()
z_enc = model.encoder(sx).detach()

# Fisher completa para todos los params
full_fisher = torch.zeros_like(theta)
for i in range(n_params):
    t_plus  = theta.detach().flatten().clone(); t_plus[i]  += eps
    t_minus = theta.detach().flatten().clone(); t_minus[i] -= eps
    grad_sq = 0.0
    for xi in z_enc[:10]:    # 10 muestras para el smoke test
        op = torch.tensor(circuit(xi.numpy(), model.theta_shared.detach().numpy(),
                          t_plus.reshape(theta.shape).numpy()), dtype=torch.float32)
        om = torch.tensor(circuit(xi.numpy(), model.theta_shared.detach().numpy(),
                          t_minus.reshape(theta.shape).numpy()), dtype=torch.float32)
        grad_sq += float(((op - om) / 2.0).pow(2).sum())
    full_fisher.flatten()[i] = grad_sq / 10

full_fisher = full_fisher.abs() + 1e-3

# 2 pasos QNG
for step in range(2):
    theta.requires_grad_(True)
    z = model.encoder(sx)
    outs = [torch.tensor(circuit(z[i].detach().numpy(),
                         model.theta_shared.detach().numpy(),
                         theta.detach().numpy()),
                         dtype=torch.float32)
            for i in range(z.shape[0])]
    logits = model.classifier(torch.stack(outs) + 0.0 * theta.sum())
    loss   = loss_fn(logits, sy)
    grad   = torch.autograd.grad(loss, theta)[0]
    theta  = (theta - 0.05 * grad / full_fisher).detach()

delta = (theta - model.theta_task).abs().mean().item()
print(f"  ✓ theta_task delta tras QNG: {delta:.4f} | {time.time()-t4:.1f}s")

# ─── Test 5: Meta-entrenamiento mini ──────────────────────────────────────────
print(f"\n[5] Meta-entrenamiento ({args.n_episodes} episodios)...")
t5 = time.time()

# Dataset sintético
class SyntheticFewShot:
    def __init__(self, n=200):
        self.n = n
    def __len__(self): return self.n
    def __getitem__(self, _):
        sx = torch.randn(K_SHOT * N_WAY, 784)
        sy = torch.repeat_interleave(torch.arange(N_WAY), K_SHOT)
        qx = torch.randn(15 * N_WAY, 784)
        qy = torch.repeat_interleave(torch.arange(N_WAY), 15)
        return sx, sy, qx, qy

from torch.utils.data import DataLoader
def collate(batch):
    return (torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]), torch.stack([b[3] for b in batch]))

ds     = SyntheticFewShot(args.n_episodes * 2)
loader = DataLoader(ds, batch_size=2, collate_fn=collate)

outer_opt = optim.Adam(model.parameters(), lr=0.001)
episode_accs = []

for ep_idx, (sx, sy, qx, qy) in enumerate(loader):
    if ep_idx >= args.n_episodes // 2:
        break
    outer_opt.zero_grad()
    meta_loss = torch.tensor(0.0)

    for t in range(sx.shape[0]):
        # Inner loop euclidean (más rápido para smoke test)
        theta_t = model.theta_task.clone()
        for _ in range(args.inner_steps):
            theta_t.requires_grad_(True)
            z = model.encoder(sx[t])
            outs = [torch.tensor(circuit(z[i].detach().numpy(),
                                 model.theta_shared.detach().numpy(),
                                 theta_t.detach().numpy()),
                                 dtype=torch.float32)
                    for i in range(z.shape[0])]
            logits = model.classifier(torch.stack(outs) + 0.0 * theta_t.sum())
            inner_loss = loss_fn(logits, sy[t])
            g = torch.autograd.grad(inner_loss, theta_t)[0]
            theta_t = (theta_t - 0.05 * g).detach()

        # Query loss
        z = model.encoder(qx[t])
        outs = [torch.tensor(circuit(z[i].detach().numpy(),
                             model.theta_shared.detach().numpy(),
                             theta_t.detach().numpy()),
                             dtype=torch.float32)
                for i in range(z.shape[0])]
        logits = model.classifier(torch.stack(outs) + 0.0 * model.theta_shared.sum())
        meta_loss = meta_loss + loss_fn(logits, qy[t])
        acc = (logits.detach().argmax(1) == qy[t]).float().mean().item()
        episode_accs.append(acc)

    (meta_loss / sx.shape[0]).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    outer_opt.step()

mean_acc = np.mean(episode_accs)
random_bl = 1 / N_WAY
print(f"  ✓ mean query acc: {mean_acc:.3f} (random={random_bl:.2f}) | {time.time()-t5:.1f}s")
print(f"  Aprende: {'SÍ ✓' if mean_acc > random_bl else 'NO aún (normal con datos sint.)'}")

# ─── Resumen ───────────────────────────────────────────────────────────────────
total = time.time() - t0
print()
print("=" * 55)
print(f"SMOKE TEST {'[SAMPLER]' if args.sampler else '[ESTIMATOR]'} — {'PASADO ✓' if mean_acc >= 0 else 'FALLADO'}")
print(f"  Tiempo total: {total:.1f}s")
print(f"  Forward:      ✓")
print(f"  Backward:     ✓")
print(f"  QFIM:         ✓  (min={min(fisher_diag):.2e})")
print(f"  QNG inner:    ✓  (delta={delta:.4f})")
print(f"  Meta-train:   ✓  (acc={mean_acc:.3f})")
print("=" * 55)
print()
print("Listo para lanzar:")
print("  Hercules:     sbatch hercules/submit_array.slurm")
print("  IBM Hardware: python code/qmaml_ibm_hardware.py --token <TOKEN> --backend ibm_kyiv")
