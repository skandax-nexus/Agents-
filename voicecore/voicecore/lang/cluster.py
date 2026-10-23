"""Unsupervised language / voice discovery.

The codec's pooled latent is a fingerprint of how an utterance sounds: speaker, prosody and
the sound inventory of its language.  Online k-means over those fingerprints discovers
clusters without any labels.  You can name a cluster once ("Hindi", "Maria") and the engine
will report it from then on.  It is a discovery mechanism, not a trained language-ID model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class VoiceClusters:
    def __init__(self, dim: int, *, max_clusters: int = 16, new_cluster_distance: float = 1.6,
                 lr: float = 0.1, split_ratio: float = 1.6, warmup: int = 3) -> None:
        self.dim = dim
        self.max_clusters = max_clusters
        self.new_cluster_distance = new_cluster_distance   # absolute floor (fingerprint units)
        self.split_ratio = split_ratio       # also split when d > split_ratio x typical within-cluster distance
        self.warmup = warmup                 # observations before the adaptive rule is trusted
        self.lr = lr
        self.centers = np.zeros((0, dim), dtype=np.float32)
        self.counts: list[int] = []
        self.names: dict[int, str] = {}
        self._n = 0
        self._avg_in = 0.0                   # running mean of accepted assignment distances
        self._n_in = 0

    def _threshold(self) -> float:
        if self._n_in < self.warmup:
            return self.new_cluster_distance
        return min(self.new_cluster_distance, self.split_ratio * self._avg_in)

    def observe(self, fingerprint: np.ndarray) -> int:
        """Assign (and adapt) a cluster for one utterance; returns its id.
        Distances are per-dimension RMS in fingerprint units (log-mel), so the threshold is interpretable."""
        z = np.asarray(fingerprint, dtype=np.float32)
        self._n += 1
        if len(self.counts) == 0:
            self.centers = z[None].copy()
            self.counts = [1]
            return 0
        d = np.linalg.norm(self.centers - z[None], axis=1) / np.sqrt(self.dim)
        k = int(d.argmin())
        if d[k] > self._threshold() and len(self.counts) < self.max_clusters:
            self.centers = np.vstack([self.centers, z[None]])
            self.counts.append(1)
            return len(self.counts) - 1
        self.centers[k] += self.lr * (z - self.centers[k])
        self.counts[k] += 1
        self._n_in += 1
        self._avg_in += (float(d[k]) - self._avg_in) / min(self._n_in, 50)
        return k

    def identify(self, fingerprint: np.ndarray) -> int | None:
        if len(self.counts) == 0:
            return None
        z = np.asarray(fingerprint, dtype=np.float32)
        return int(np.linalg.norm(self.centers - z[None], axis=1).argmin())

    def name(self, cluster: int, label: str) -> None:
        self.names[int(cluster)] = label

    def label(self, cluster: int | None) -> str:
        if cluster is None:
            return "unknown"
        return self.names.get(int(cluster), f"cluster-{cluster}")

    def summary(self) -> list[dict]:
        return [{"id": i, "name": self.label(i), "utterances": c} for i, c in enumerate(self.counts)]

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"dim": self.dim, "centers": self.centers.tolist(), "counts": self.counts,
                                          "names": self.names, "n": self._n, "max_clusters": self.max_clusters,
                                          "new_cluster_distance": self.new_cluster_distance, "lr": self.lr,
                                          "split_ratio": self.split_ratio, "warmup": self.warmup,
                                          "avg_in": self._avg_in, "n_in": self._n_in}))

    @classmethod
    def load(cls, path: Path) -> "VoiceClusters":
        d = json.loads(Path(path).read_text())
        vc = cls(d["dim"], max_clusters=d["max_clusters"], new_cluster_distance=d["new_cluster_distance"], lr=d["lr"],
                 split_ratio=d.get("split_ratio", 1.6), warmup=d.get("warmup", 3))
        vc._avg_in, vc._n_in = d.get("avg_in", 0.0), d.get("n_in", 0)
        vc.centers = np.array(d["centers"], dtype=np.float32).reshape(-1, d["dim"])
        vc.counts = list(d["counts"])
        vc.names = {int(k): v for k, v in d["names"].items()}
        vc._n = d["n"]
        return vc
