# nilearn-aal-viewer

> **Interactive brain activation viewer with AAL3 anatomical labeling**  
> Click any voxel → get the region name and MNI coordinates instantly.

Built on [nilearn](https://nilearn.github.io) · No server required · Single self-contained HTML output

---

## What it does

`show_brain_viewer` wraps nilearn's `view_img` to add:

- A **fixed title bar** displaying the contrast and statistical threshold
- A **click badge** returning the AAL3 region name and MNI coordinates of any clicked voxel
- A **fully portable HTML file** that can be shared by email or embedded in a report

![demo](example/screenshot.png)

*Related article: [Interactive Brain Activation Maps with Anatomical Labeling](https://medium.com/@f.kucharczak) — Medium*

---

## Requirements

```
nibabel >= 3.2
nilearn >= 0.10
numpy  >= 1.22
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

```python
from viewer import build_aal_lookup, lut_to_json, show_brain_viewer
from nilearn import datasets
import json

# 1 — Build the lookup table once per session
lut      = build_aal_lookup(aal_path="AAL3v1_1mm.nii.gz", step_mm=2)
lut_json = lut_to_json(lut)

# 2 — Generate the viewer
show_brain_viewer(
    img            = tmap_nii,            # your NIfTI t-map
    threshold      = 3.5,                 # display threshold
    title          = "C > B",             # short contrast label
    contrast_label = "p<0.001 uncorrected",
    fname          = "viewer_C_vs_B.html",
    out_dir        = "./outputs",
    lut_json       = lut_json,
    step           = 2,
    bg_img         = datasets.load_mni152_template(),
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
| `step` | int | `2` | LUT resolution in mm — must match `build_aal_lookup` |
| `bg_img` | NIfTI | MNI152 | Background anatomical image |
| `cmap` | str | `'cold_hot'` | Colormap (`'hot'`, `'inferno'`, etc.) |
| `opacity` | float | `0.8` | Overlay opacity [0–1] |
| `height` | int | `520` | Jupyter IFrame height in px |

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
