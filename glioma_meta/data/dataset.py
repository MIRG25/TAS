"""Four-channel NIfTI dataset and reproducible image preprocessing."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .labels import MODALITIES


def _load_image(path: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    image = nib.load(path)
    array = image.get_fdata(dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI at {path}; got {array.shape}")
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0), spacing


def _load_volume(path: str) -> np.ndarray:
    return _load_image(path)[0]


def _z_score_nonzero(volume: np.ndarray) -> np.ndarray:
    mask = volume != 0
    if mask.sum() < 16:
        return volume.astype(np.float32)
    values = volume[mask]
    clipped = np.clip(values, np.percentile(values, 0.5), np.percentile(values, 99.5))
    result = np.zeros_like(volume, dtype=np.float32)
    result[mask] = (clipped - clipped.mean()) / max(float(clipped.std()), 1e-6)
    return result


def _crop_bounds(mask: np.ndarray, shape: tuple[int, int, int], margin: int = 12) -> tuple[slice, slice, slice]:
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        # The whole brain contains robust context if a tumour mask is unavailable/empty.
        return tuple(slice(0, dim) for dim in shape)  # type: ignore[return-value]
    low = np.maximum(coords.min(axis=0) - margin, 0)
    high = np.minimum(coords.max(axis=0) + margin + 1, shape)
    return tuple(slice(int(left), int(right)) for left, right in zip(low, high))  # type: ignore[return-value]


def _fixed_window_slices(
    center_xyz: np.ndarray,
    window_shape_xyz: np.ndarray,
    image_shape_xyz: tuple[int, int, int],
) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice], np.ndarray, np.ndarray]:
    """Return source/destination slices for a centred fixed window with padding."""
    window_shape_xyz = np.asarray(window_shape_xyz, dtype=int)
    if np.any(window_shape_xyz <= 0):
        raise ValueError(f"Fixed ROI dimensions must be positive; got {window_shape_xyz.tolist()}")
    image_shape = np.asarray(image_shape_xyz, dtype=int)
    start = np.floor(np.asarray(center_xyz) - (window_shape_xyz - 1) / 2.0).astype(int)
    # When an ROI axis spans the complete source axis, retain that complete
    # axis instead of shifting it according to the tumour centre.
    full_axis_start = -((window_shape_xyz - image_shape) // 2)
    start = np.where(window_shape_xyz >= image_shape, full_axis_start, start)
    end = start + window_shape_xyz
    source_low = np.maximum(start, 0)
    source_high = np.minimum(end, image_shape)
    destination_low = source_low - start
    destination_high = destination_low + np.maximum(source_high - source_low, 0)
    source = tuple(slice(int(low), int(high)) for low, high in zip(source_low, source_high))
    destination = tuple(
        slice(int(low), int(high)) for low, high in zip(destination_low, destination_high)
    )
    return source, destination, start, end  # type: ignore[return-value]


def _crop_pad_fixed(
    volume: np.ndarray,
    source: tuple[slice, slice, slice],
    destination: tuple[slice, slice, slice],
    window_shape_xyz: np.ndarray,
) -> np.ndarray:
    result = np.zeros(tuple(int(value) for value in window_shape_xyz), dtype=volume.dtype)
    result[destination] = volume[source]
    return result


def _online_crop(
    image: np.ndarray,
    tumour_mask: np.ndarray,
    output_shape_dhw: Sequence[int],
    random_crop: bool,
    min_tumor_retention: float,
    attempts: int,
    tumor_guided_probability: float = 1.0,
    min_brain_fraction: float = 0.0,
    tumor_center_jitter_voxels: int = 0,
    audit: dict | None = None,
) -> np.ndarray:
    """Crop aligned modalities together from a cached full MRI volume.

    Training uses a mixture of tumour-guided crops and foreground-constrained
    whole-brain crops. Validation/test are deterministic and centred on the
    tumour when a mask is available, otherwise on the complete cached volume.
    The mask is used only to choose a crop and is never returned as an input
    channel. Optional audit metadata describes the selected crop without any
    extra random draws or changes to the sampling policy.
    """
    output_shape = np.asarray(tuple(int(value) for value in output_shape_dhw), dtype=int)
    spatial_shape = np.asarray(image.shape[1:], dtype=int)
    if image.ndim != 4 or tumour_mask.shape != tuple(spatial_shape):
        raise ValueError("Cached MRI and tumour mask must have aligned [C,D,H,W]/[D,H,W] shapes")
    if np.any(output_shape > spatial_shape):
        raise ValueError(
            f"Online crop {output_shape.tolist()} exceeds cached spatial shape {spatial_shape.tolist()}"
        )
    if not 0.0 <= min_tumor_retention <= 1.0:
        raise ValueError("data.random_crop.min_tumor_retention must be in [0, 1]")
    if not 0.0 <= tumor_guided_probability <= 1.0:
        raise ValueError("data.random_crop.tumor_guided_probability must be in [0, 1]")
    if not 0.0 <= min_brain_fraction <= 1.0:
        raise ValueError("data.random_crop.min_brain_fraction must be in [0, 1]")
    if int(tumor_center_jitter_voxels) < 0:
        raise ValueError("data.random_crop.tumor_center_jitter_voxels must be non-negative")

    maximum_start = spatial_shape - output_shape
    centered_start = maximum_start // 2
    tumour_voxels = float(np.count_nonzero(tumour_mask))
    brain_mask = np.any(image != 0, axis=0)
    crop_voxels = float(np.prod(output_shape))

    def crop_at(start: np.ndarray) -> tuple[np.ndarray, float, float]:
        slices = tuple(slice(int(left), int(left + size)) for left, size in zip(start, output_shape))
        retained = float(np.count_nonzero(tumour_mask[slices]) / max(tumour_voxels, 1.0))
        brain_fraction = float(np.count_nonzero(brain_mask[slices]) / crop_voxels)
        return image[(slice(None), *slices)], retained, brain_fraction

    def contiguous_float32(crop: np.ndarray) -> np.ndarray:
        # Cast only the selected 128^3 crop. This preserves the memory-saving
        # benefit of float16 full-volume memmaps.
        return np.array(crop, dtype=np.float32, copy=True, order="C")

    def finish(start: np.ndarray, branch: str, fallback: bool = False) -> np.ndarray:
        crop, retention, brain_fraction = crop_at(start)
        if audit is not None:
            audit.update({
                "branch": branch,
                "fallback": bool(fallback),
                "tumor_voxels": int(tumour_voxels),
                "tumor_retention": retention if tumour_voxels > 0 else float("nan"),
                "min_tumor_retention": float(min_tumor_retention),
                "brain_fraction": brain_fraction,
                "start_d": int(start[0]),
                "start_h": int(start[1]),
                "start_w": int(start[2]),
            })
        return contiguous_float32(crop)

    def start_around_tumour(jitter: int) -> np.ndarray:
        coordinates = np.argwhere(tumour_mask)
        centre = (coordinates.min(axis=0) + coordinates.max(axis=0)) / 2.0
        if jitter > 0:
            offset = np.asarray(
                [int(torch.randint(-jitter, jitter + 1, (1,)).item()) for _ in range(3)],
                dtype=float,
            )
            centre += offset
        start = np.rint(centre - (output_shape - 1) / 2.0).astype(int)
        return np.clip(start, 0, maximum_start)

    if not random_crop or not np.any(maximum_start > 0):
        deterministic_start = start_around_tumour(0) if tumour_voxels > 0 else centered_start
        return finish(deterministic_start, "deterministic")

    use_tumour_guidance = tumour_voxels > 0 and bool(
        torch.rand(1).item() < tumor_guided_probability
    )
    n_attempts = max(int(attempts), 1)
    if use_tumour_guidance:
        best_start = start_around_tumour(0)
        _, best_retention, _ = crop_at(best_start)
        for _ in range(n_attempts):
            start = start_around_tumour(int(tumor_center_jitter_voxels))
            _, retention, _ = crop_at(start)
            if retention >= min_tumor_retention:
                return finish(start, "tumor_guided")
            if retention > best_retention:
                best_start, best_retention = start, retention
        return finish(best_start, "tumor_guided", fallback=True)

    best_start = centered_start
    _, _, best_brain_fraction = crop_at(best_start)
    for _ in range(n_attempts):
        start = np.asarray(
            [int(torch.randint(0, int(limit) + 1, (1,)).item()) for limit in maximum_start],
            dtype=int,
        )
        _, _, brain_fraction = crop_at(start)
        if brain_fraction >= min_brain_fraction:
            return finish(start, "random_brain")
        if brain_fraction > best_brain_fraction:
            best_start, best_brain_fraction = start, brain_fraction
    return finish(best_start, "random_brain", fallback=True)


def load_preprocess_case(
    row: pd.Series,
    crop_shape: Sequence[int],
    use_tumor_mask_for_crop: bool = True,
    roi_mode: str = "full_volume",
    roi_size_mm: Sequence[float] = (176.0, 176.0, 155.0),
    return_metadata: bool = False,
    return_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict] | tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict]:
    """Return a four-channel [C, D, H, W] MRI tensor and optional ROI audit data."""
    loaded = [_load_image(row[f"{name.lower()}_path"]) for name in MODALITIES]
    volumes = [item[0] for item in loaded]
    spacings = [item[1] for item in loaded]
    if len({volume.shape for volume in volumes}) != 1:
        raise ValueError(f"Modality shapes differ for patient {row.patient_id}")
    if not all(np.allclose(spacing, spacings[0], atol=1e-4) for spacing in spacings[1:]):
        raise ValueError(f"Modality voxel spacings differ for patient {row.patient_id}")
    reference = volumes[0]
    tumour_path = str(row.get("tumor_mask_path", ""))
    audit: dict = {
        "patient_id": str(row.patient_id),
        "roi_mode": roi_mode,
        "input_x_vox": int(reference.shape[0]),
        "input_y_vox": int(reference.shape[1]),
        "input_z_vox": int(reference.shape[2]),
        "spacing_x_mm": float(spacings[0][0]),
        "spacing_y_mm": float(spacings[0][1]),
        "spacing_z_mm": float(spacings[0][2]),
    }
    if roi_mode == "full_volume":
        cropped_volumes = volumes
        if tumour_path and Path(tumour_path).is_file():
            tumour_mask, mask_spacing = _load_image(tumour_path)
            tumour_mask = tumour_mask > 0
            if tumour_mask.shape != reference.shape or not np.allclose(
                mask_spacing, spacings[0], atol=1e-4
            ):
                raise ValueError(f"Tumour mask grid differs from MRI grid for patient {row.patient_id}")
        elif use_tumor_mask_for_crop:
            raise ValueError(f"Tumour mask is required for guided cropping: {row.patient_id}")
        else:
            tumour_mask = np.zeros(reference.shape, dtype=bool)
        cropped_mask = tumour_mask
        audit.update(
            {
                "tumor_voxels": int(np.count_nonzero(cropped_mask)),
                "full_volume_unchanged": True,
                "window_padded": False,
            }
        )
    elif roi_mode == "fixed_physical":
        if not use_tumor_mask_for_crop:
            raise ValueError("fixed_physical ROI requires use_tumor_mask_for_crop: true")
        if not tumour_path or not Path(tumour_path).is_file():
            raise ValueError(f"Tumour mask is required for fixed ROI localization: {row.patient_id}")
        tumour_mask, mask_spacing = _load_image(tumour_path)
        tumour_mask = tumour_mask > 0
        if tumour_mask.shape != reference.shape or not np.allclose(mask_spacing, spacings[0], atol=1e-4):
            raise ValueError(f"Tumour mask grid differs from MRI grid for patient {row.patient_id}")
        coordinates = np.argwhere(tumour_mask)
        if not len(coordinates):
            raise ValueError(f"Tumour mask is empty for patient {row.patient_id}")
        bbox_low = coordinates.min(axis=0)
        bbox_high = coordinates.max(axis=0)
        center_xyz = (bbox_low + bbox_high) / 2.0
        roi_size_mm_xyz = np.asarray(tuple(float(value) for value in roi_size_mm), dtype=float)
        if roi_size_mm_xyz.shape != (3,) or np.any(roi_size_mm_xyz <= 0):
            raise ValueError("data.roi_size_mm must contain three positive x/y/z dimensions")
        spacing_xyz = np.asarray(spacings[0], dtype=float)
        window_shape_xyz = np.maximum(np.rint(roi_size_mm_xyz / spacing_xyz).astype(int), 1)
        source, destination, window_start, window_end = _fixed_window_slices(
            center_xyz, window_shape_xyz, reference.shape
        )
        cropped_volumes = [
            _crop_pad_fixed(volume, source, destination, window_shape_xyz) for volume in volumes
        ]
        cropped_mask = _crop_pad_fixed(tumour_mask, source, destination, window_shape_xyz)
        retained_fraction = float(cropped_mask.sum() / max(float(tumour_mask.sum()), 1.0))
        audit.update(
            {
                **{f"roi_{axis}_mm": float(value) for axis, value in zip("xyz", roi_size_mm_xyz)},
                **{f"spacing_{axis}_mm": float(value) for axis, value in zip("xyz", spacing_xyz)},
                **{f"window_{axis}_vox": int(value) for axis, value in zip("xyz", window_shape_xyz)},
                **{f"bbox_{axis}_vox": int(value) for axis, value in zip("xyz", bbox_high - bbox_low + 1)},
                "tumor_retained_fraction": retained_fraction,
                "tumor_fully_contained": bool(retained_fraction >= 1.0 - 1e-8),
                "window_padded": bool(np.any(window_start < 0) or np.any(window_end > np.asarray(reference.shape))),
            }
        )
    elif roi_mode == "resized_bbox":
        crop_mask = _load_volume(tumour_path) > 0 if use_tumor_mask_for_crop and tumour_path and Path(tumour_path).is_file() else reference != 0
        bounds = _crop_bounds(crop_mask, reference.shape)
        cropped_volumes = [volume[bounds] for volume in volumes]
        cropped_mask = crop_mask[bounds]
    else:
        raise ValueError("data.roi_mode must be full_volume, fixed_physical, or resized_bbox")

    # NIfTI is x/y/z; model convention is D/H/W. Interpolation changes only
    # voxel spacing within the already-fixed physical FOV, never tumour scale.
    stacked = np.stack([_z_score_nonzero(volume) for volume in cropped_volumes], axis=0)
    stacked = np.transpose(stacked, (0, 3, 2, 1))
    output_shape = tuple(int(value) for value in crop_shape)
    transposed_mask = np.transpose(cropped_mask.astype(bool), (2, 1, 0))
    if roi_mode == "full_volume":
        if tuple(stacked.shape[1:]) != output_shape:
            expected_xyz = tuple(reversed(output_shape))
            raise ValueError(
                f"Full-volume shape mismatch for patient {row.patient_id}: "
                f"source XYZ={reference.shape}, expected XYZ={expected_xyz}. "
                "Full-volume mode never resizes or pads; audit this dataset and choose an explicit "
                "harmonization policy before caching."
            )
        result = np.ascontiguousarray(stacked, dtype=np.float32)
        result_mask = np.ascontiguousarray(transposed_mask)
    else:
        tensor = torch.from_numpy(stacked).unsqueeze(0)
        if tuple(tensor.shape[2:]) != output_shape:
            tensor = F.interpolate(tensor, size=output_shape, mode="trilinear", align_corners=False)
        result = tensor.squeeze(0).numpy().astype(np.float32, copy=False)
        mask_tensor = torch.from_numpy(transposed_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        if tuple(mask_tensor.shape[2:]) != output_shape:
            mask_tensor = F.interpolate(mask_tensor, size=output_shape, mode="nearest")
        result_mask = mask_tensor.squeeze(0).squeeze(0).numpy() > 0.5
    audit.update(
        {
            "cache_d_vox": int(result.shape[1]),
            "cache_h_vox": int(result.shape[2]),
            "cache_w_vox": int(result.shape[3]),
        }
    )
    if return_mask and return_metadata:
        return result, result_mask, audit
    if return_mask:
        return result, result_mask
    return (result, audit) if return_metadata else result


class GliomaMRIDataset(Dataset):
    """Manifest-backed MRI dataset. Labels are returned with explicit masks."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        patient_ids: Sequence[str],
        crop_shape: Sequence[int],
        cache_shape: Sequence[int] | None = None,
        cache_dir: str | Path | None = None,
        use_tumor_mask_for_crop: bool = True,
        roi_mode: str = "full_volume",
        roi_size_mm: Sequence[float] = (176.0, 176.0, 155.0),
        random_crop: bool = False,
        min_tumor_retention: float = 0.70,
        random_crop_attempts: int = 32,
        tumor_guided_probability: float = 1.0,
        min_brain_fraction: float = 0.0,
        tumor_center_jitter_voxels: int = 0,
        label_availability: Mapping[str, Mapping[str, bool]] | None = None,
        included_tasks: Sequence[str] | None = None,
        audit_crops: bool = False,
    ) -> None:
        self.frame = manifest.set_index("patient_id", drop=False).loc[list(patient_ids)].reset_index(drop=True)
        self.crop_shape = tuple(int(v) for v in crop_shape)
        self.cache_shape = tuple(int(v) for v in (cache_shape or crop_shape))
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_tumor_mask_for_crop = use_tumor_mask_for_crop
        self.roi_mode = roi_mode
        self.roi_size_mm = tuple(float(value) for value in roi_size_mm)
        self.random_crop = bool(random_crop)
        self.min_tumor_retention = float(min_tumor_retention)
        self.random_crop_attempts = int(random_crop_attempts)
        self.tumor_guided_probability = float(tumor_guided_probability)
        self.min_brain_fraction = float(min_brain_fraction)
        self.tumor_center_jitter_voxels = int(tumor_center_jitter_voxels)
        self.audit_crops = bool(audit_crops)
        self.label_availability = label_availability or {}
        self.included_tasks = tuple(
            dict.fromkeys(included_tasks or ("idh", "who", "survival"))
        )
        unknown_tasks = set(self.included_tasks) - {"idh", "who", "survival"}
        if unknown_tasks:
            raise ValueError(f"Unknown included_tasks: {sorted(unknown_tasks)}")
        if self.label_availability:
            for task in ("idh", "who", "survival"):
                self.frame[f"{task}_mask"] = self.frame["patient_id"].map(
                    lambda patient_id: bool(self.label_availability.get(str(patient_id), {}).get(task, False))
                )

    def __len__(self) -> int:
        return len(self.frame)

    def _image(self, row: pd.Series, audit: dict | None = None) -> np.ndarray:
        if self.cache_dir:
            cache_file = self.cache_dir / f"{row.patient_id}.npy"
            mask_file = self.cache_dir / f"{row.patient_id}_mask.npy"
            if cache_file.is_file():
                # Memory mapping avoids reading the entire cached ROI
                # when only one online 128^3 crop is consumed.
                image = np.load(cache_file, mmap_mode="r")
                expected_shape = (4, *self.cache_shape)
                if image.shape != expected_shape:
                    raise ValueError(
                        f"Cached image shape {image.shape} does not match configured {expected_shape}: {cache_file}"
                    )
                if not mask_file.is_file():
                    raise ValueError(f"Cached tumour mask is missing: {mask_file}")
                tumour_mask = np.load(mask_file, mmap_mode="r").astype(bool, copy=False)
                return _online_crop(
                    image,
                    tumour_mask,
                    self.crop_shape,
                    self.random_crop,
                    self.min_tumor_retention,
                    self.random_crop_attempts,
                    self.tumor_guided_probability,
                    self.min_brain_fraction,
                    self.tumor_center_jitter_voxels,
                    audit=audit,
                )
        image, tumour_mask = load_preprocess_case(
            row,
            self.cache_shape,
            self.use_tumor_mask_for_crop,
            roi_mode=self.roi_mode,
            roi_size_mm=self.roi_size_mm,
            return_mask=True,
        )
        return _online_crop(
            image,
            tumour_mask,
            self.crop_shape,
            self.random_crop,
            self.min_tumor_retention,
            self.random_crop_attempts,
            self.tumor_guided_probability,
            self.min_brain_fraction,
            self.tumor_center_jitter_voxels,
            audit=audit,
        )

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        patient_id = str(row.patient_id)
        availability = self.label_availability.get(patient_id, {})
        crop_audit = {} if self.audit_crops else None
        item = {
            "image": torch.from_numpy(self._image(row, audit=crop_audit)),
            "patient_id": patient_id,
        }
        if crop_audit is not None:
            item["crop_audit"] = crop_audit
        if "idh" in self.included_tasks:
            item.update(
                {
                    "idh": torch.tensor(int(row.idh_label), dtype=torch.long),
                    "idh_mask": torch.tensor(
                        bool(row.get("idh_mask", availability.get("idh", True)))
                    ),
                }
            )
        if "who" in self.included_tasks:
            item.update(
                {
                    "who": torch.tensor(int(row.who_label), dtype=torch.long),
                    "who_mask": torch.tensor(
                        bool(row.get("who_mask", availability.get("who", True)))
                    ),
                }
            )
        if "survival" in self.included_tasks:
            item.update(
                {
                    "survival_time": torch.tensor(
                        float(row.survival_time), dtype=torch.float32
                    ),
                    "survival_event": torch.tensor(
                        float(row.survival_event), dtype=torch.float32
                    ),
                    "survival_mask": torch.tensor(
                        bool(row.get("survival_mask", availability.get("survival", True)))
                    ),
                }
            )
        return item
