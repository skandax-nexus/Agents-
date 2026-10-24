"""Unsupervised voice / language discovery.

Each heard utterance yields a fingerprint (see ``VoiceCore.fingerprint``).  Fingerprints are kept,
and clusters are found by a deterministic batch k-means over all of them (``refit``), either on
request (``voicecore langs --refit K``) or during self-training.  Between refits every new
utterance is assigned to its nearest centroid.  Name a cluster once ("Hindi", "Maria") and the
engine reports it from then on; names survive refits by centroid matching.

Measured honestly: on 0.3 s recordings of six speakers the trained codec's fingerprint reaches
~0.8 nearest-centroid accuracy with labelled centroids and ~0.7 k-means purity at k = 6. It is a
discovery aid, not a speaker-ID or language-ID model; longer utterances separate better.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def _kmeans(X: np.ndarray, k: int, seed: int = 0, iters: int = 60, restarts: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """k-means++ with restarts; returns (assignments, centroids) of the lowest-inertia run."""
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for r in range(restarts):
        rs = np.random.default_rng(seed + r)
        C = X[[rs.integers(len(X))]]
        while len(C) < k:                                   # k-means++ seeding
            d2 = ((X[:, None] - C[None]) ** 2).sum(-1).min(1)
            probs = d2 / d2.sum() if d2.sum() > 0 else np.full(len(X), 1 / len(X))
            C = np.vstack([C, X[rs.choice(len(X), p=probs)]])
        for _ in range(iters):
            a = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
            newC = np.array([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = float(((X - C[a]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, a, C)
    return best[1], best[2]


class VoiceClusters:
    def __init__(self, dim: int, *, max_fingerprints: int = 5000) -> None:
        self.dim = dim
        self.max_fingerprints = max_fingerprints
        self.centers = np.zeros((0, dim), dtype=np.float32)
        self.counts: list[int] = []
        self.names: dict[int, str] = {}
        self.fingerprints = np.zeros((0, dim), dtype=np.float32)
        self.assignments: list[int] = []
        self._mean = np.zeros(dim, dtype=np.float32)
        self._scale = np.ones(dim, dtype=np.float32)

    # -- standardisation (fitted at refit time so distances weigh every dimension equally)
    def _z(self, x: np.ndarray) -> np.ndarray:
        return (x - self._mean) / self._scale

    # -- online path --------------------------------------------------------------------------------
    def observe(self, fingerprint: np.ndarray) -> int:
        """Remember an utterance's fingerprint and assign it to the nearest known cluster (0 until the first refit)."""
        x = np.asarray(fingerprint, dtype=np.float32).reshape(1, -1)
        if len(self.fingerprints) >= self.max_fingerprints:      # reservoir: drop the oldest
            self.fingerprints = self.fingerprints[1:]
            self.assignments = self.assignments[1:]
        self.fingerprints = np.vstack([self.fingerprints, x])
        if len(self.counts) == 0:
            self.centers = x.copy()
            self.counts = [1]
            self.assignments.append(0)
            return 0
        k = self.identify(x[0])
        self.counts[k] += 1
        self.assignments.append(k)
        return k

    def identify(self, fingerprint: np.ndarray) -> int | None:
        if len(self.counts) == 0:
            return None
        z = self._z(np.asarray(fingerprint, dtype=np.float32))
        return int(np.linalg.norm(self._z(self.centers) - z[None], axis=1).argmin())

    # -- batch discovery ------------------------------------------------------------------------------
    def refit(self, k: int | None = None, *, seed: int = 0) -> dict:
        """Re-cluster every stored fingerprint into k groups (default: keep the current number, at least 2).
        Names follow the centroids they were closest to."""
        n = len(self.fingerprints)
        if n < 2:
            return {"clusters": len(self.counts), "fingerprints": n, "note": "need at least 2 utterances"}
        k = max(1, min(int(k or max(2, len(self.counts))), n))
        X = self.fingerprints
        self._mean = X.mean(0)
        self._scale = X.std(0) + 1e-6
        Z = self._z(X)
        a, C = _kmeans(Z, k, seed=seed)
        new_centers = C * self._scale + self._mean
        # carry names over to the nearest new centroid (one name per centroid)
        new_names: dict[int, str] = {}
        if self.names and len(self.centers):
            oldZ = self._z(self.centers)
            for old_id, name in self.names.items():
                if old_id < len(oldZ):
                    j = int(np.linalg.norm(C - oldZ[old_id][None], axis=1).argmin())
                    new_names.setdefault(j, name)
        self.centers = new_centers.astype(np.float32)
        self.counts = [int((a == j).sum()) for j in range(k)]
        self.assignments = [int(v) for v in a]
        self.names = new_names
        return {"clusters": k, "fingerprints": n, "sizes": self.counts}

    # -- naming / reporting ----------------------------------------------------------------------------
    def name(self, cluster: int, label: str) -> None:
        self.names[int(cluster)] = label

    def label(self, cluster: int | None) -> str:
        if cluster is None:
            return "unknown"
        return self.names.get(int(cluster), f"cluster-{cluster}")

    def summary(self) -> list[dict]:
        return [{"id": i, "name": self.label(i), "utterances": c} for i, c in enumerate(self.counts)]

    def save(self, path: Path) -> None:
        path = Path(path)
        npy = path.with_suffix(".npy")
        tmp_npy = npy.with_suffix(".npy.tmp")
        with open(tmp_npy, "wb") as fh:
            np.save(fh, self.fingerprints)
        os.replace(tmp_npy, npy)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"dim": self.dim, "centers": self.centers.tolist(), "counts": self.counts,
                                   "names": self.names, "assignments": self.assignments,
                                   "mean": self._mean.tolist(), "scale": self._scale.tolist(),
                                   "max_fingerprints": self.max_fingerprints}))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> VoiceClusters:
        path = Path(path)
        d = json.loads(path.read_text())
        vc = cls(d["dim"], max_fingerprints=d.get("max_fingerprints", 5000))
        vc.centers = np.array(d["centers"], dtype=np.float32).reshape(-1, d["dim"])
        vc.counts = list(d["counts"])
        vc.names = {int(k): v for k, v in d["names"].items()}
        vc.assignments = list(d.get("assignments", []))
        vc._mean = np.array(d.get("mean", [0.0] * d["dim"]), dtype=np.float32)
        vc._scale = np.array(d.get("scale", [1.0] * d["dim"]), dtype=np.float32)
        npy = path.with_suffix(".npy")
        if npy.exists():
            vc.fingerprints = np.load(npy).astype(np.float32).reshape(-1, d["dim"])
        return vc
