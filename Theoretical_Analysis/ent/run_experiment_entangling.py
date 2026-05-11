import argparse
import csv
import json
import os
import time

import numpy as np
import pennylane as qml

import circuits_pennylane as pl_circ


SAMPLE_SIZES   = [1500, 2500, 5000, 7500]
ANSATZ_ORDER   = ["Cand-NN", "Cand-CB", "Cand-AA"]
N_QUBITS       = pl_circ.N_QUBITS

ANSATZ_SEED_OFFSET = {
    "Cand-NN":  4001,
    "Cand-CB":  4007,
    "Cand-AA":  4013,
}


def build_state_fn(name):
    n_params, apply_fn = pl_circ.ANSATZ_REGISTRY[name]()
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="numpy")
    def state_qnode(weights):
        apply_fn(weights)
        return qml.state()

    def run(weights_batch):
        out = np.empty((weights_batch.shape[0], 2 ** N_QUBITS), dtype=np.complex128)
        for k in range(weights_batch.shape[0]):
            out[k] = np.asarray(state_qnode(weights_batch[k]))
        return out

    return n_params, run


def meyer_wallach_via_pennylane(states, n_qubits):
    B = states.shape[0]
    all_wires = list(range(n_qubits))
    Q = np.empty(B, dtype=np.float64)
    for b in range(B):
        psi = states[b]
        rho = np.outer(psi, psi.conj())
        purity_sum = 0.0
        for j in range(n_qubits):
            wires_to_trace = [w for w in all_wires if w != j]
            rho_j = qml.math.partial_trace(rho, wires_to_trace, c_dtype="complex128")
            purity_sum += float(np.trace(rho_j @ rho_j).real)
        Q[b] = 2.0 * (1.0 - purity_sum / n_qubits)
    return np.clip(Q, 0.0, 1.0)


def sample_pqc_mw(state_fn, n_params, n_samples, rng):
    thetas = rng.uniform(0.0, 2.0 * np.pi, size=(n_samples, n_params))
    states = state_fn(thetas)
    return meyer_wallach_via_pennylane(states, N_QUBITS)


def run(out_dir, seed):
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for name in ANSATZ_ORDER:
        n_params, state_fn = build_state_fn(name)
        rng = np.random.default_rng(int(seed) * 1000003 + ANSATZ_SEED_OFFSET[name])

        print(f"[{name}]  n_params={n_params}", flush=True)
        t0 = time.time()
        Q_full = sample_pqc_mw(state_fn, n_params, max(SAMPLE_SIZES), rng)
        t_sample = time.time() - t0
        print(f"   MW values computed in {t_sample:.2f}s  "
              f"(mean Q over {max(SAMPLE_SIZES)} states = {Q_full.mean():.5f})",
              flush=True)

        for M in SAMPLE_SIZES:
            sub = Q_full[:M]
            ent = float(sub.mean())
            ent_se = float(sub.std(ddof=1) / np.sqrt(M))
            row = {
                "ansatz":    name,
                "n_params":  int(n_params),
                "n_samples": int(M),
                "ent_mean":  ent,
                "ent_sem":   ent_se,
                "q_min":     float(sub.min()),
                "q_max":     float(sub.max()),
                "q_std":     float(sub.std(ddof=1)),
            }
            rows.append(row)
            print(f"   M={M:5d}  Ent = {ent:.6f}  (SEM = {ent_se:.6f})", flush=True)
        print()

    csv_path  = os.path.join(out_dir, "results.csv")
    json_path = os.path.join(out_dir, "results.json")

    fieldnames = ["ansatz", "n_params", "n_samples",
                  "ent_mean", "ent_sem",
                  "q_min", "q_max", "q_std"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})

    meta = {
        "seed": int(seed),
        "n_qubits": int(N_QUBITS),
        "sample_sizes": SAMPLE_SIZES,
        "ansatz_order": ANSATZ_ORDER,
        "ansatz_seed_offsets": ANSATZ_SEED_OFFSET,
        "rows": rows,
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"{'='*82}\nResults  (Ent = mean Meyer-Wallach Q;  Q in [0, 1];  larger = more entangled)")
    print(f"{'='*82}\n")
    print(f"{'ansatz':<10s} {'#params':>8s} ", *[f"{M:>10d}" for M in SAMPLE_SIZES])
    print("-" * (10 + 9 + 11 * len(SAMPLE_SIZES)))
    grid = {(r["ansatz"], r["n_samples"]): r["ent_mean"] for r in rows}
    pmap = {r["ansatz"]: r["n_params"] for r in rows}
    for a in ANSATZ_ORDER:
        cells = [grid[(a, M)] for M in SAMPLE_SIZES]
        print(f"{a:<10s} {pmap[a]:>8d} ", *[f"{v:>10.6f}" for v in cells])

    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="./output")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.out_dir, args.seed)
