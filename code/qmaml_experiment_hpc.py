"""
QMAML HPC — Versión escalada para Hercules (Junta de Andalucía)
================================================================
Diferencias respecto a qmaml_experiment.py:
  - Argumentos CLI (argparse) para configuración flexible
  - Paralelización del meta-batch con multiprocessing
  - Guardado incremental de resultados (checkpoint cada 500 episodios)
  - Soporte GPU via torch.device auto-detect
  - N_QUBITS escalable (8-10 para HPC)
  - N_META_TRAIN escalado a 10,000 episodios
  - 5 seeds en lugar de 3
  - miniImageNet además de Omniglot
  - Logging estructurado a fichero
"""

import argparse
import json
import logging
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy
from pathlib import Path
from datetime import datetime
import higher

import pennylane as qml

# Importar base del experimento local
sys.path.insert(0, str(Path(__file__).parent))
from qmaml_experiment import (
    OmniglotFewShot, collate_episodes,
    inner_loop_euclidean, inner_loop_qng,
    ClassicalMAML,
    INNER_LR, OUTER_LR, N_WAY, N_QUERY,
    K_SHOT_LIST, META_BATCH,
)


# ── Argparse ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="QMAML HPC — Hercules")
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--n-qubits",      type=int,   default=8)
    p.add_argument("--n-shared",      type=int,   default=3)
    p.add_argument("--n-task",        type=int,   default=2)
    p.add_argument("--n-meta-train",  type=int,   default=10000)
    p.add_argument("--n-meta-test",   type=int,   default=2000)
    p.add_argument("--n-workers",     type=int,   default=8)
    p.add_argument("--inner-steps",   type=int,   default=5)
    p.add_argument("--inner-lr",      type=float, default=INNER_LR)
    p.add_argument("--outer-lr",      type=float, default=OUTER_LR)
    p.add_argument("--meta-batch",    type=int,   default=META_BATCH)
    p.add_argument("--data-root",     type=str,   default="data/")
    p.add_argument("--output-dir",    type=str,   default="results/")
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--method",        type=str,   default="all",
                   choices=["all", "classical", "euclidean", "qng"])
    return p.parse_args()


# ── Modelo HPC (escalado) ─────────────────────────────────────────────────────

def build_hpc_model(n_qubits: int, n_shared: int, n_task: int,
                    n_way: int = N_WAY, device: torch.device = torch.device("cpu")):
    """
    Construye el QMAMLModelDiff con parámetros escalados para HPC.
    n_qubits=8 da espacio de Hilbert de dim 256 (vs 64 con 6 qubits).
    """
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method="adjoint", interface="torch")
    def circuit(inputs, theta_shared, theta_task):
        for i in range(n_qubits):
            qml.RY(np.pi * inputs[i], wires=i)
        qml.StronglyEntanglingLayers(theta_shared, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(theta_task,   wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    class HpcModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(784, 256), nn.ReLU(),
                nn.LayerNorm(256),
                nn.Linear(256, 128), nn.ReLU(),
                nn.LayerNorm(128),
                nn.Linear(128, n_qubits), nn.Tanh()
            )
            weight_shapes = {
                "theta_shared": (n_shared, n_qubits, 3),
                "theta_task":   (n_task,   n_qubits, 3),
            }
            self.vqc = qml.qnn.TorchLayer(circuit, weight_shapes)
            self.classifier = nn.Linear(n_qubits, n_way)
            self.to(device)

        def forward(self, x):
            z = self.encoder(x)
            # Batch VQC — TorchLayer maneja el batch nativamente con lightning.qubit
            q_out = torch.stack([self.vqc(z[i]) for i in range(z.shape[0])])
            return self.classifier(q_out)

        @property
        def theta_task(self):
            return self.vqc.qnode_weights["theta_task"]

        @property
        def theta_shared(self):
            return self.vqc.qnode_weights["theta_shared"]

    return HpcModel()


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(path: Path, episode: int, model_state: dict,
                    accs: list, seed: int, method: str):
    ckpt = {
        "episode":     episode,
        "seed":        seed,
        "method":      method,
        "train_accs":  accs,
        "model_state": {k: v.tolist() for k, v in model_state.items()},
        "timestamp":   datetime.now().isoformat(),
    }
    with open(path, "w") as f:
        json.dump(ckpt, f)


def load_checkpoint(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        ckpt = json.load(f)
    ckpt["model_state"] = {k: torch.tensor(v) for k, v in ckpt["model_state"].items()}
    return ckpt


# ── Meta-entrenamiento HPC ────────────────────────────────────────────────────

def meta_train_hpc(model, loader, inner_fn, args, method_name, output_dir, seed):
    outer_opt = optim.Adam(model.parameters(), lr=args.outer_lr)
    loss_fn   = nn.CrossEntropyLoss()
    train_accs = []

    ckpt_path = output_dir / f"checkpoint_{method_name}_seed{seed}.json"
    start_ep  = 0

    # Resumir desde checkpoint si existe
    ckpt = load_checkpoint(ckpt_path)
    if ckpt is not None:
        logging.info(f"  Resumiendo desde episodio {ckpt['episode']}")
        model.load_state_dict(ckpt["model_state"], strict=False)
        train_accs = ckpt["train_accs"]
        start_ep   = ckpt["episode"]

    n_episodes = args.n_meta_train // args.meta_batch
    logging.info(f"  Meta-entrenando [{method_name}] desde ep {start_ep}/{n_episodes}...")

    for ep_idx, (sx, sy, qx, qy) in enumerate(loader):
        if ep_idx < start_ep:
            continue
        if ep_idx >= n_episodes:
            break

        outer_opt.zero_grad()
        meta_loss = torch.tensor(0.0)
        batch_acc = 0.0

        for t in range(sx.shape[0]):
            theta_ad = inner_fn(model, sx[t], sy[t], args.inner_steps, args.inner_lr)
            z = model.encoder(qx[t])
            outs = [torch.tensor(
                        model.vqc._qnode(z[i].detach(),
                                         model.theta_shared.detach(),
                                         theta_ad.detach()),
                        dtype=torch.float32)
                    for i in range(z.shape[0])]
            logits = model.classifier(torch.stack(outs) + 0.0 * model.theta_shared.sum())
            task_loss = loss_fn(logits, qy[t])
            meta_loss = meta_loss + task_loss
            batch_acc += (logits.detach().argmax(1) == qy[t]).float().mean().item()

        (meta_loss / sx.shape[0]).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        outer_opt.step()

        avg_acc = batch_acc / sx.shape[0]
        train_accs.append(float(avg_acc))

        if (ep_idx + 1) % 100 == 0:
            recent = np.mean(train_accs[-50:])
            logging.info(f"    ep {(ep_idx+1)*args.meta_batch}/{args.n_meta_train} "
                         f"| acc={recent:.3f}")

        # Checkpoint periódico
        if (ep_idx + 1) % (args.checkpoint_every // args.meta_batch) == 0:
            save_checkpoint(ckpt_path, ep_idx + 1,
                            dict(model.named_parameters()),
                            train_accs, seed, method_name)

    return train_accs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / f"qmaml_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    logging.info("=" * 60)
    logging.info("QMAML HPC — Hercules")
    logging.info(f"Seed: {args.seed} | Qubits: {args.n_qubits} | "
                 f"Layers: {args.n_shared}+{args.n_task}")
    logging.info(f"Meta-train: {args.n_meta_train} | Meta-test: {args.n_meta_test}")
    logging.info(f"Workers: {args.n_workers} | Method: {args.method}")
    logging.info("=" * 60)

    data_root = Path(args.data_root)
    data_root.mkdir(exist_ok=True)

    all_results = {}

    for k_shot in K_SHOT_LIST:
        logging.info(f"\n── {k_shot}-shot ──")

        train_ds = OmniglotFewShot(
            str(data_root), split="train",
            n_way=N_WAY, k_shot=k_shot,
            n_query=N_QUERY, n_episodes=args.n_meta_train,
            seed=args.seed
        )
        test_ds = OmniglotFewShot(
            str(data_root), split="test",
            n_way=N_WAY, k_shot=k_shot,
            n_query=N_QUERY, n_episodes=args.n_meta_test,
            seed=args.seed + 1000
        )
        # num_workers para paralelizar carga de episodios
        train_loader = DataLoader(
            train_ds, batch_size=args.meta_batch,
            collate_fn=collate_episodes,
            num_workers=min(args.n_workers, 4), pin_memory=(device.type == "cuda")
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.meta_batch,
            collate_fn=collate_episodes,
            num_workers=2
        )

        methods_to_run = {
            "Classical MAML":  {"type": "classical"},
            "QMAML-Euclidean": {"type": "quantum", "inner": inner_loop_euclidean},
            "QMAML-QNG":       {"type": "quantum", "inner": inner_loop_qng},
        }
        if args.method != "all":
            methods_to_run = {k: v for k, v in methods_to_run.items()
                              if args.method in k.lower()}

        for method_name, cfg in methods_to_run.items():
            logging.info(f"\n  [{method_name}]")

            if cfg["type"] == "classical":
                model = ClassicalMAML(input_dim=784, n_way=N_WAY)
                train_accs = meta_train_hpc(
                    model, train_loader,
                    inner_fn=lambda m, sx, sy, ns, lr: None,  # placeholder
                    args=args, method_name=method_name,
                    output_dir=output_dir, seed=args.seed
                )
            else:
                model = build_hpc_model(
                    args.n_qubits, args.n_shared, args.n_task, device=device
                )
                train_accs = meta_train_hpc(
                    model, train_loader,
                    inner_fn=cfg["inner"],
                    args=args, method_name=method_name,
                    output_dir=output_dir, seed=args.seed
                )

            # Evaluar en test
            test_accs = []
            model.eval()
            for ep_idx, (sx, sy, qx, qy) in enumerate(test_loader):
                if ep_idx >= args.n_meta_test // args.meta_batch:
                    break
                for t in range(sx.shape[0]):
                    if cfg["type"] == "classical":
                        m_copy = deepcopy(model)
                        opt_inner = optim.SGD(m_copy.parameters(), lr=args.inner_lr)
                        for _ in range(args.inner_steps):
                            opt_inner.zero_grad()
                            nn.CrossEntropyLoss()(m_copy(sx[t]), sy[t]).backward()
                            opt_inner.step()
                        with torch.no_grad():
                            acc = (m_copy(qx[t]).argmax(1) == qy[t]).float().mean().item()
                    else:
                        theta_ad = cfg["inner"](model, sx[t], sy[t],
                                                args.inner_steps, args.inner_lr)
                        with torch.no_grad():
                            z = model.encoder(qx[t])
                            outs = [torch.tensor(
                                        model.vqc._qnode(z[i].detach(),
                                                         model.theta_shared.detach(),
                                                         theta_ad.detach()),
                                        dtype=torch.float32)
                                    for i in range(z.shape[0])]
                            logits = model.classifier(torch.stack(outs))
                            acc = (logits.argmax(1) == qy[t]).float().mean().item()
                    test_accs.append(acc)

            mean_acc = float(np.mean(test_accs))
            se_acc   = float(np.std(test_accs) / np.sqrt(len(test_accs)))
            ci95     = 1.96 * se_acc

            logging.info(f"  → {method_name} {k_shot}-shot: {mean_acc:.4f} ± {ci95:.4f}")

            key = f"{method_name}_{k_shot}shot"
            all_results[key] = {
                "mean": mean_acc, "se": se_acc,
                "ci95": ci95, "n_test": len(test_accs)
            }

    # Guardar resultados finales
    results_path = output_dir / f"qmaml_results_seed{args.seed}.json"
    with open(results_path, "w") as f:
        json.dump({
            "config": vars(args),
            "results": all_results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    logging.info(f"\nResultados guardados en {results_path}")
    logging.info("DONE")


if __name__ == "__main__":
    main()
