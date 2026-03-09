"""
aal_lookup.py
-------------
Build a voxel-to-region lookup table from an AAL3v1 NIfTI atlas.

The LUT maps MNI coordinate strings "x_y_z" to AAL3 region names.
It is built once per session and serialized to JSON for injection
into the interactive HTML viewer.

Usage
-----
    from viewer.aal_lookup import build_aal_lookup
    import json

    lut = build_aal_lookup(aal_path="AAL3v1_1mm.nii.gz", step_mm=2)
    lut_json = json.dumps(lut, separators=(',', ':'))
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np


def build_aal_lookup(
    aal_path: str | Path,
    step_mm: int = 2,
    verbose: bool = True,
) -> dict[str, str]:
    """
    Build a MNI-coordinate → AAL3 region name lookup table.

    Parameters
    ----------
    aal_path : str or Path
        Path to the AAL3v1 NIfTI file (e.g. ``AAL3v1_1mm.nii.gz``).
        The companion label file (``.txt``) must be in the same directory.
        Several filename variants are tried automatically.
    step_mm : int, optional
        Resampling resolution in mm. Default 2 mm — good trade-off between
        speed (~5 s), memory (~3 MB), and spatial accuracy for typical
        PET/fMRI cluster sizes. Use 1 for single-voxel precision.
    verbose : bool, optional
        Print progress information. Default True.

    Returns
    -------
    dict[str, str]
        Dictionary mapping ``"x_y_z"`` MNI coordinate strings to region
        names (e.g. ``"-60_-50_14": "Temporal_Sup_L"``).

    Raises
    ------
    FileNotFoundError
        If the AAL3 NIfTI or label file cannot be found.

    Notes
    -----
    The resampling preserves the original affine's translation vector so
    that negative MNI coordinates (bounding box origin ~[-90, -126, -72])
    are correctly handled. This is the most common failure point when
    building atlas LUTs.
    """
    from nilearn.image import resample_img

    aal_path = Path(aal_path)
    if not aal_path.exists():
        raise FileNotFoundError(f"AAL3 NIfTI not found: {aal_path}")

    # ── Locate companion label file ───────────────────────────────────────
    stem = aal_path.stem.split('.')[0]   # handle .nii.gz double extension
    label_candidates = [
        aal_path.with_suffix('').with_suffix('.txt'),
        aal_path.with_suffix('.txt'),
        aal_path.parent / f"{stem}.txt",
        aal_path.parent / 'AAL3v1_1mm.txt',
    ]
    label_file = next((p for p in label_candidates if p.exists()), None)
    if label_file is None:
        raise FileNotFoundError(
            f"AAL3 label file not found. Tried:\n" +
            "\n".join(f"  {p}" for p in label_candidates)
        )

    # ── Load region labels ────────────────────────────────────────────────
    labels: dict[int, str] = {}
    with open(label_file, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    labels[int(parts[0])] = parts[1]
                except ValueError:
                    pass

    if verbose:
        print(f"  AAL3 labels loaded : {len(labels)} regions ({label_file.name})")

    # ── Resample atlas ────────────────────────────────────────────────────
    aal_img = nib.load(str(aal_path))
    orig_affine = aal_img.affine

    # Preserve origin — critical for correct negative MNI coordinates
    target_affine = np.diag([step_mm, step_mm, step_mm, 1]).astype(float)
    target_affine[:3, 3] = orig_affine[:3, 3]

    aal_res = resample_img(aal_img, target_affine=target_affine,
                           interpolation='nearest')
    data   = aal_res.get_fdata().astype(int)
    affine = aal_res.affine

    # ── Build lookup table ────────────────────────────────────────────────
    lut: dict[str, str] = {}
    for vox in np.argwhere(data > 0):
        mni = nib.affines.apply_affine(affine, vox)
        x, y, z = (int(round(v)) for v in mni)
        region_idx = int(data[tuple(vox)])
        lut[f'{x}_{y}_{z}'] = labels.get(region_idx, f'Region_{region_idx}')

    if verbose:
        print(f"  LUT built : {len(lut)} voxels, "
              f"{len(set(lut.values()))} regions, "
              f"resolution {step_mm} mm")

    return lut


def lut_to_json(lut: dict[str, str]) -> str:
    """Serialize the LUT to a compact JSON string for HTML injection."""
    return json.dumps(lut, separators=(',', ':'))
