"""Cached whole-brain MRI crops for self-supervised foundation pretraining."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FoundationMRIDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        crop_shape: Sequence[int] = (128, 128, 128),
        tumor_guided_probability: float = 0.5,
        tumor_center_jitter_voxels: int = 24,
        min_brain_fraction: float = 0.20,
        random_crop_attempts: int = 8,
        flip_probability: float = 0.5,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        self.frame = manifest[manifest["split"] == split].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"No cases in split={split}")
        self.split = split
        self.crop_shape = tuple(int(value) for value in crop_shape)
        if len(self.crop_shape) != 3 or any(value <= 0 for value in self.crop_shape):
            raise ValueError("crop_shape must contain three positive D/H/W values")
        self.tumor_guided_probability = float(tumor_guided_probability)
        self.tumor_center_jitter_voxels = int(tumor_center_jitter_voxels)
        self.min_brain_fraction = float(min_brain_fraction)
        self.random_crop_attempts = max(int(random_crop_attempts), 1)
        self.flip_probability = float(flip_probability)
        if not 0.0 <= self.tumor_guided_probability <= 1.0:
            raise ValueError("tumor_guided_probability must lie in [0, 1]")
        if not 0.0 <= self.min_brain_fraction <= 1.0:
            raise ValueError("min_brain_fraction must lie in [0, 1]")

    def __len__(self) -> int:
        return len(self.frame)

    def _bbox(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray] | None:
        columns = [
            "tumor_bbox_d0",
            "tumor_bbox_h0",
            "tumor_bbox_w0",
            "tumor_bbox_d1",
            "tumor_bbox_h1",
            "tumor_bbox_w1",
        ]
        values = pd.to_numeric(row[columns], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            return None
        low, high = values[:3], values[3:]
        if np.any(high <= low):
            return None
        return low, high

    def _crop_at(self, image: np.ndarray, start: np.ndarray) -> np.ndarray:
        slices = tuple(
            slice(int(left), int(left + size)) for left, size in zip(start, self.crop_shape)
        )
        return np.asarray(image[(slice(None), *slices)], dtype=np.float32)

    def _center_start(self, spatial_shape: np.ndarray) -> np.ndarray:
        return np.maximum((spatial_shape - np.asarray(self.crop_shape)) // 2, 0)

    def _tumor_start(
        self, spatial_shape: np.ndarray, bbox: tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        low, high = bbox
        center = (low + high - 1.0) / 2.0
        jitter = np.random.randint(
            -self.tumor_center_jitter_voxels,
            self.tumor_center_jitter_voxels + 1,
            size=3,
        )
        start = np.rint(center - (np.asarray(self.crop_shape) - 1) / 2.0 + jitter).astype(int)
        return np.clip(start, 0, spatial_shape - np.asarray(self.crop_shape))

    def _random_brain_crop(self, image: np.ndarray, spatial_shape: np.ndarray) -> np.ndarray:
        maximum = spatial_shape - np.asarray(self.crop_shape)
        best_crop = None
        best_fraction = -1.0
        for _ in range(self.random_crop_attempts):
            start = np.asarray(
                [np.random.randint(0, int(value) + 1) for value in maximum], dtype=int
            )
            crop = self._crop_at(image, start)
            foreground_fraction = float(np.any(crop != 0, axis=0).mean())
            if foreground_fraction >= self.min_brain_fraction:
                return crop
            if foreground_fraction > best_fraction:
                best_crop, best_fraction = crop, foreground_fraction
        if best_crop is not None:
            return best_crop
        return self._crop_at(image, self._center_start(spatial_shape))

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        cache_path = Path(str(row.cache_path))
        image = np.load(cache_path, mmap_mode="r")
        if image.ndim != 4 or image.shape[0] != 4:
            raise ValueError(f"Expected [4,D,H,W] cache at {cache_path}; got {image.shape}")
        spatial_shape = np.asarray(image.shape[1:], dtype=int)
        if np.any(np.asarray(self.crop_shape) > spatial_shape):
            raise ValueError(f"Crop {self.crop_shape} exceeds cache shape {tuple(spatial_shape)}")

        if self.split == "validation":
            crop = self._crop_at(image, self._center_start(spatial_shape))
        else:
            bbox = self._bbox(row)
            use_tumor = bbox is not None and np.random.random() < self.tumor_guided_probability
            crop = (
                self._crop_at(image, self._tumor_start(spatial_shape, bbox))
                if use_tumor
                else self._random_brain_crop(image, spatial_shape)
            )
            # Preserve superior-inferior orientation; augment only the two in-plane axes.
            if np.random.random() < self.flip_probability:
                crop = np.flip(crop, axis=2)
            if np.random.random() < self.flip_probability:
                crop = np.flip(crop, axis=3)

        crop = np.ascontiguousarray(crop, dtype=np.float32)
        return {
            "image": torch.from_numpy(crop),
            "cache_key": str(row.cache_key),
            "dataset": str(row.dataset),
            "patient_id": str(row.patient_id),
        }


def seed_foundation_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
