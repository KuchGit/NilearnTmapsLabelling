# NilearnTmapsLabelling

> **Interactive brain activation viewer with AAL3 anatomical labeling**  
> Click any voxel → get the region name and MNI coordinates instantly.  
> Click any cluster row → the viewer jumps to that peak.

Built on [nilearn](https://nilearn.github.io) · No server required · Single self-contained HTML output

---

## What it does

`show_brain_viewer` wraps nilearn's `view_img` to add:

- A **fixed title bar** displaying the contrast and statistical threshold
- A **click badge** returning the AAL3 region name and MNI coordinates of any clicked voxel
- An optional **cluster table** appended below the viewer, generated from `nilearn.reporting.get_clusters_table`:
  - AAL3 region name auto-filled on the peak voxel of each cluster
  - Sortable columns (click any header)
  - Click a row → the viewer crosshair jumps to that cluster's peak
- A **fully portable HTML file** that can be shared by email or embedded in a report

![demo](example/demo.gif)

*Related article: [Interactive Brain Activation Maps with Anatomical Labeling](https://medium.com/@f.kucharczak) — Medium*

---

## Requirements

```
nibabel  >= 3.2
nilearn  >= 0.10
numpy    >= 1.22
scipy    >= 1.7
```

Install:

```bash
pip install -r requirements.txt
```

**AAL3v1 atlas** — not included (third-party data).  
Download `AAL3v1_1mm.nii.gz` and its companion `.txt` label file from:  
→ https://www.gin.cnrs.fr/en/tools/aal/

---

## Quick start

### Viewer only

```python
from viewer import build_aal_lookup, lut_to_json, show_brain_viewer
from nilearn import datasets

# Build the lookup table once per session
lut      = build_aal_lookup(aal_path="AAL3v1_1mm.nii.gz", step_mm=2)
lut_json = lut_to_json(lut)

show_brain_viewer(
    img            = tmap_nii,
    threshold      = 3.5,
    title          = "C > B",
    contrast_label = "p<0.001 uncorrected",
    fname          = "viewer_C_vs_B.html",
    out_dir        = "./outputs",
    lut_json       = lut_json,
    step           = 2,
    bg_img         = datasets.load_mni152_template(),
)
```

### Viewer + interactive cluster table

```python
from nilearn.reporting import get_clusters_table

tbl = get_clusters_table(tmap_nii, stat_threshold=3.5, cluster_threshold=50)

show_brain_viewer(
    img                  = tmap_nii,
    threshold            = 3.5,
    title                = "C > B",
    contrast_label       = "p<0.001 uncorrected",
    fname                = "viewer_C_vs_B.html",
    out_dir              = "./outputs",
    lut_json             = lut_json,
    lut                  = lut,           # enables ROI column + row navigation
    step                 = 2,
    bg_img               = datasets.load_mni152_template(),
    clusters_table       = tbl,
    clusters_table_title = "Clusters C > B — p<0.001",
)
```

Output: `outputs/viewer_C_vs_B_aal.html` — open in any browser.

See [`example/demo.ipynb`](example/demo.ipynb) for a full walkthrough with synthetic data.

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `img` | NIfTI | — | Statistical map (t-map, z-map) |
| `threshold` | float | — | Display threshold (absolute value) |
| `title` | str | — | Short contrast label for the title bar |
| `contrast_label` | str | — | Statistical qualifier (threshold, model) |
| `fname` | str | — | Output filename (`.html`) |
| `out_dir` | str/Path | — | Output directory |
| `lut_json` | str | — | Serialized AAL LUT from `lut_to_json()` |
| `lut` | dict | `None` | Python LUT dict — enables ROI column and row click navigation |
| `step` | int | `2` | LUT resolution in mm — must match `build_aal_lookup` |
| `bg_img` | NIfTI | MNI152 | Background anatomical image |
| `cmap` | str | `'cold_hot'` | Colormap (`'hot'`, `'inferno'`, etc.) |
| `opacity` | float | `0.8` | Overlay opacity [0–1] |
| `height` | int | `520` | Jupyter IFrame height in px |
| `clusters_table` | DataFrame | `None` | Output of `get_clusters_table()` — appended below viewer |
| `clusters_table_title` | str | `'Cluster table'` | Title shown above the table |

### Displaying only extreme values

To show only the most significant voxels, threshold the map before passing it:

```python
import nibabel as nib
import numpy as np

data = tmap_nii.get_fdata().copy()
data[np.abs(data) < 4.5] = 0   # keep only |t| > 4.5
img_extreme = nib.Nifti1Image(data, tmap_nii.affine)

show_brain_viewer(img=img_extreme, threshold=3.5, ...)
```

---

## Repository structure

```
nilearn-aal-viewer/
├── viewer/
│   ├── __init__.py
│   ├── aal_lookup.py      # build_aal_lookup(), lut_to_json()
│   └── brain_viewer.py    # show_brain_viewer()
├── example/
│   └── demo.ipynb         # full demo with synthetic data
├── requirements.txt
├── LICENSE                # MIT
└── .gitignore
```

---

## Citation / atlas reference

If you use this viewer in a publication, please cite the AAL3 atlas:

> Rolls ET, Huang CC, Lin CP, Feng J, Joliot M. (2020).  
> *Automated anatomical labelling atlas 3.*  
> NeuroImage, 206, 116189. https://doi.org/10.1016/j.neuroimage.2019.116189

---

## License

MIT © [Florentin Kucharczak](https://f.kucharczak.github.io)
