"""Memmap loader over packed token streams.

Sequences are fixed, non-overlapping windows of the flat token stream. Each
epoch visits every window exactly once in a seeded shuffle, which keeps the
token-for-token comparison between arms exact and reproducible.
"""
import json, os
import numpy as np
import torch


class PackedDataset:
    def __init__(self, bin_path, seq_len, dtype=None, mask_path=None, meta_path=None):
        if meta_path is None:
            meta_path = bin_path.replace(".bin", ".meta.json")
        self.meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        self.dtype = np.dtype(dtype or self.meta.get("dtype", "uint16"))
        self.bin_path = bin_path
        self.seq_len = seq_len
        self.tokens = np.memmap(bin_path, dtype=self.dtype, mode="r")
        self.n_tokens = len(self.tokens)
        # windows of seq_len+1 tokens (inputs + shifted targets)
        self.n_seqs = (self.n_tokens - 1) // seq_len
        self.mask = None
        if mask_path and os.path.exists(mask_path):
            self.mask = np.memmap(mask_path, dtype=np.uint8, mode="r")
            assert len(self.mask) == self.n_tokens, \
                f"mask/token length mismatch: {len(self.mask)} vs {self.n_tokens}"

    def __len__(self):
        return self.n_seqs

    def get(self, idx):
        s = idx * self.seq_len
        w = np.asarray(self.tokens[s:s + self.seq_len + 1], dtype=np.int64)
        m = None
        if self.mask is not None:
            m = np.asarray(self.mask[s + 1:s + self.seq_len + 1], dtype=np.int64)
        return w, m

    def __repr__(self):
        return (f"PackedDataset({os.path.basename(self.bin_path)}, "
                f"{self.n_tokens/1e6:.1f}M tokens, {self.n_seqs} seqs of {self.seq_len}, "
                f"mask={'yes' if self.mask is not None else 'no'})")


class SequenceSampler:
    """Infinite, epoch-aware seeded shuffle over window indices."""

    def __init__(self, n_seqs, seed=0, start_step=0, batch_size=1, limit=None):
        self.n = n_seqs if limit is None else min(n_seqs, limit)
        self.seed = seed
        self.batch_size = batch_size
        self.pos = 0
        self.epoch = 0
        self._order = self._perm(0)
        # fast-forward for resume
        for _ in range(start_step * batch_size):
            self._next_idx()

    def _perm(self, epoch):
        g = np.random.default_rng(self.seed + 1000003 * epoch)
        return g.permutation(self.n)

    def _next_idx(self):
        if self.pos >= self.n:
            self.epoch += 1
            self._order = self._perm(self.epoch)
            self.pos = 0
        i = self._order[self.pos]
        self.pos += 1
        return int(i)

    def next_batch(self, bs=None):
        bs = bs or self.batch_size
        return [self._next_idx() for _ in range(bs)]


def make_batch(ds, idxs, device, use_mask=True):
    xs, ys, ms = [], [], []
    for i in idxs:
        w, m = ds.get(i)
        xs.append(w[:-1]); ys.append(w[1:])
        if m is not None:
            ms.append(m)
    x = torch.from_numpy(np.stack(xs))
    y = torch.from_numpy(np.stack(ys))
    mask = torch.from_numpy(np.stack(ms)) if (ms and use_mask) else None
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    if mask is not None:
        mask = mask.to(device, non_blocking=True)
    return x, y, mask
