import argparse
import csv
import json
import os
import time

import numpy as np
import pennylane as qml
from scipy.special import rel_entr

import circuits_pennylane as pl_circ

SAMPLE_SIZES   = [1500, 2500, 5000, 7500]
N_BINS         = 75
EPSILON        = 1e-10
ANSATZ_ORDER   = ["Ref-NN", "Ref-CB", "Ref-AA", "Cand-NN", "Cand-CB", "Cand-AA"]
HILBERT_DIM    = 2 ** pl_circ.N_QUBITS

EXPR_UPPER_BOUND = (HILBERT_DIM - 1) * np.log(N_BINS)

ANSATZ_SEED_OFFSET = {
    "Ref-NN":  101,
    "Ref-CB":  211,
    "Ref-AA":  307,
    "Cand-NN": 401,
    "Cand-CB": 503,
    "Cand-AA": 601,
}
HAAR_SEED_OFFSET = 7919


def build_state_fn(name):
    n_params, apply_fn = pl_circ.ANSATZ_REGISTRY[name]()
    dev = qml.device("default.qubit", wires=pl_circ.N_QUBITS)

    @qml.qnode(dev, interface="numpy")
    def state_qnode(weights):
        apply_fn(weights)
        return qml.state()

    def run(weights_batch):
        out = np.empty((weights_batch.shape[0], HILBERT_DIM), dtype=np.complex128)
        for k in range(weights_batch.shape[0]):
            out[k] = np.asarray(state_qnode(weights_batch[k]))
        return out

    return n_params, run


def sample_pqc_fidelities(state_fn, n_params, n_samples, rng):
    thetas = rng.uniform(0.0, 2.0 * np.pi, size=(n_samples, n_params))
    phis   = rng.uniform(0.0, 2.0 * np.pi, size=(n_samples, n_params))
    states_a = state_fn(thetas)
    states_b = state_fn(phis)
    inner = np.einsum('bi,bi->b', states_a.conj(), states_b)
    return np.abs(inner) ** 2


def sample_haar_states(n_samples, dim, rng):
    z = rng.normal(size=(n_samples, dim)) + 1j * rng.normal(size=(n_samples, dim))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return z


def sample_haar_fidelities(n_samples, rng):
    states_a = sample_haar_states(n_samples, HILBERT_DIM, rng)
    states_b = sample_haar_states(n_samples, HILBERT_DIM, rng)
    inner = np.einsum('bi,bi->b', states_a.conj(), states_b)
    return np.abs(inner) ** 2


def kl_divergence_from_fids(fids_pqc, fids_haar):
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    c_p, _ = np.histogram(fids_pqc,  bins=edges, density=False)
    c_q, _ = np.histogram(fids_haar, bins=edges, density=False)

    p = c_p.astype(np.float64) + EPSILON
    q = c_q.astype(np.float64) + EPSILON
    p /= p.sum()
    q /= q.sum()

    return float(np.sum(rel_entr(p, q)))


def run(out_dir, seed):
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Haar baseline]  sampling {max(SAMPLE_SIZES)} pairs ...", flush=True)
    rng_haar = np.random.default_rng(int(seed) * 1000033 + HAAR_SEED_OFFSET)
    t0 = time.time()
    haar_fids_full = sample_haar_fidelities(max(SAMPLE_SIZES), rng_haar)
    print(f"   done in {time.time()-t0:.2f}s  "
          f"(mean F = {haar_fids_full.mean():.5f}, "
          f"theoretical = {1.0/(HILBERT_DIM+1):.5f})\n", flush=True)

    rows = []
    for name in ANSATZ_ORDER:
        n_params, state_fn = build_state_fn(name)
        rng_pqc = np.random.default_rng(int(seed) * 1000003 + ANSATZ_SEED_OFFSET[name])

        print(f"[{name}]  n_params={n_params}", flush=True)
        t0 = time.time()
        pqc_fids_full = sample_pqc_fidelities(state_fn, n_params, max(SAMPLE_SIZES), rng_pqc)
        t_sample = time.time() - t0
        print(f"   PQC fidelities sampled in {t_sample:.2f}s  "
              f"(mean F = {pqc_fids_full.mean():.5f})", flush=True)

        for N in SAMPLE_SIZES:
            kl = kl_divergence_from_fids(pqc_fids_full[:N], haar_fids_full[:N])
            row = {
                "ansatz": name,
                "n_params": int(n_params),
                "n_samples": int(N),
                "kl_divergence": kl,
                "pqc_fid_mean": float(pqc_fids_full[:N].mean()),
                "pqc_fid_std":  float(pqc_fids_full[:N].std()),
                "haar_fid_mean": float(haar_fids_full[:N].mean()),
                "haar_fid_std":  float(haar_fids_full[:N].std()),
            }
            rows.append(row)
            print(f"   N={N:5d}  KL = {kl:.6f}", flush=True)
        print()

    csv_path  = os.path.join(out_dir, "results.csv")
    json_path = os.path.join(out_dir, "results.json")

    fieldnames = ["ansatz", "n_params", "n_samples",
                  "kl_divergence",
                  "pqc_fid_mean", "pqc_fid_std",
                  "haar_fid_mean", "haar_fid_std"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})

    meta = {
        "seed": int(seed),
        "n_qubits": int(pl_circ.N_QUBITS),
        "hilbert_dim": int(HILBERT_DIM),
        "n_bins": int(N_BINS),
        "epsilon": EPSILON,
        "expr_upper_bound": EXPR_UPPER_BOUND,
        "haar_fid_theoretical_mean": 1.0 / (HILBERT_DIM + 1),
        "sample_sizes": SAMPLE_SIZES,
        "ansatz_order": ANSATZ_ORDER,
        "ansatz_seed_offsets": ANSATZ_SEED_OFFSET,
        "haar_seed_offset": HAAR_SEED_OFFSET,
        "rows": rows,
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"{'='*82}\nResults  (KL divergence, lower = closer to Haar = more expressible)")
    print(f"Theoretical upper bound = (DIM-1) * ln(n_bins) = 63 * ln(75) "
          f"= {EXPR_UPPER_BOUND:.2f}\n{'='*82}\n")
    print(f"{'ansatz':<10s} {'#params':>8s} ", *[f"{N:>10d}" for N in SAMPLE_SIZES])
    print("-" * (10 + 9 + 11 * len(SAMPLE_SIZES)))
    grid = {(r["ansatz"], r["n_samples"]): r["kl_divergence"] for r in rows}
    pmap = {r["ansatz"]: r["n_params"] for r in rows}
    for a in ANSATZ_ORDER:
        cells = [grid[(a, N)] for N in SAMPLE_SIZES]
        print(f"{a:<10s} {pmap[a]:>8d} ", *[f"{v:>10.4f}" for v in cells])

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
