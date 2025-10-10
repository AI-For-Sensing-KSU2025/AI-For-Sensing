import os, glob, random
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
from torch.utils.data import Dataset

def _read_bin_f32(path: str, shape: Tuple[int, ...]) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(
            f"File {path} has {arr.size} floats but expected {expected} for shape {shape}. "
            f"Fix heatmap_shape or verify file."
        )
    return arr.reshape(shape)

def compute_channel_stats(paths: List[str], heatmap_shape: Tuple[int, ...], sample_k: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    picks = paths if len(paths) <= sample_k else random.sample(paths, sample_k)
    C = heatmap_shape[0] if len(heatmap_shape) == 3 else 1
    sums = np.zeros(C, dtype=np.float64)
    sqs  = np.zeros(C, dtype=np.float64)
    count = 0
    for p in picks:
        x = _read_bin_f32(p, heatmap_shape)
        if x.ndim == 2:
            x = x[None, ...]                # (1,H,W)
        x = x.reshape(x.shape[0], -1)       # (C, H*W)
        sums += x.mean(axis=1)
        sqs  += x.var(axis=1)
        count += 1
    mean = sums / max(count, 1)
    std  = np.sqrt(sqs / max(count, 1) + 1e-8)
    return mean.astype(np.float32), std.astype(np.float32)

class ColoradarHeatmapDataset(Dataset):
    """
    Loads ColoRadar heatmaps (.bin) as (C,H,W) tensors .
    """
    def __init__(
        self,
        data_root: str,
        sequences: Optional[List[str]] = None,
        pattern: str = "radar_heatmaps/**/*.bin",
        heatmap_shape: Tuple[int, ...] = (2, 256, 256),
        normalization: str = "zscore",
        stats: Optional[Dict[str, List[float]]] = None
    ):
        super().__init__()
        self.data_root = data_root
        self.heatmap_shape = heatmap_shape
        self.normalization = normalization.lower()

        # Collect .bin files
        seqs = sequences if sequences is not None else [
            d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))
        ]
        self.samples = []  # list[(path, seq_name)]
        for s in seqs:
            base = os.path.join(data_root, s)
            for p in glob.glob(os.path.join(base, pattern), recursive=True):
                self.samples.append((p, s))
        if not self.samples:
            raise FileNotFoundError(f"No .bin heatmaps found under {data_root} with pattern '{pattern}'.")

        # Normalization stats
        if self.normalization == "zscore":
            if stats is None:
                paths = [p for p, _ in self.samples]
                mean, std = compute_channel_stats(paths, heatmap_shape)
                self.stats = {"mean": mean.tolist(), "std": std.tolist()}
            else:
                self.stats = stats
        elif self.normalization == "minmax":
            C = heatmap_shape[0] if len(heatmap_shape) == 3 else 1
            self.stats = {"min": [0.0]*C, "max": [1.0]*C}
        else:
            self.stats = None

    def __len__(self):
        return len(self.samples)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalization == "zscore" and self.stats is not None:
            mean = torch.tensor(self.stats["mean"], dtype=x.dtype, device=x.device).view(-1, 1, 1)
            std  = torch.tensor(self.stats["std"],  dtype=x.dtype, device=x.device).view(-1, 1, 1)
            return (x - mean) / (std + 1e-6)
        if self.normalization == "minmax" and self.stats is not None:
            mn = torch.tensor(self.stats["min"], dtype=x.dtype, device=x.device).view(-1, 1, 1)
            mx = torch.tensor(self.stats["max"], dtype=x.dtype, device=x.device).view(-1, 1, 1)
            return (x - mn) / (mx - mn + 1e-6)
        return x

    def __getitem__(self, idx):
        path, seq = self.samples[idx]
        x = torch.from_numpy(_read_bin_f32(path, self.heatmap_shape)).float()
        if x.ndim == 2:
            x = x.unsqueeze(0)  # (H,W) -> (1,H,W)
        x = self._normalize(x)
        return {"image": x, "path": path, "sequence": seq}
