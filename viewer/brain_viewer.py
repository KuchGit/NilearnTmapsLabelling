"""
brain_viewer.py
---------------
Generate an interactive HTML brain activation viewer with AAL3 anatomical
labeling on click, built on top of nilearn's ``view_img``.

Usage
-----
    from viewer.aal_lookup import build_aal_lookup, lut_to_json
    from viewer.brain_viewer import show_brain_viewer
    from nilearn import datasets
    import nibabel as nib

    lut      = build_aal_lookup("AAL3v1_1mm.nii.gz", step_mm=2)
    lut_json = lut_to_json(lut)
    bg_img   = datasets.load_mni152_template()

    show_brain_viewer(
        img            = tmap_nii,
        threshold      = 3.5,
        title          = "C > B",
        contrast_label = "p<0.001 uncorrected",
        fname          = "viewer_C_vs_B.html",
        out_dir        = "./outputs",
        lut_json       = lut_json,
        step           = 2,
        bg_img         = bg_img,
    )
"""

from __future__ import annotations

from pathlib import Path

from nilearn import plotting


# ── JavaScript / CSS injected into the HTML output ───────────────────────

_BADGE_CSS = """
<style>
/* ── fixed title bar ── */
#pet-title-bar {
  position: fixed;
  top: 0; left: 0; right: 0;
  display: flex; align-items: baseline; gap: 12px;
  padding: 8px 20px 6px 20px;
  background: rgba(10, 20, 50, 0.92);
  backdrop-filter: blur(4px);
  z-index: 99998;
  font-family: 'Helvetica Neue', Arial, sans-serif;
  border-bottom: 1px solid rgba(255,255,255,0.08);
}
#pet-title-main {
  color: #e8eaf6;
  font-size: 15px;
  font-weight: 600;
  letter-spacing: 0.03em;
}
#pet-title-sub {
  color: #7986cb;
  font-size: 12px;
  font-weight: 400;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
/* ── AAL region badge ── */
#aal-badge {
  position: fixed;
  top: 36px; left: 0; right: 0;
  display: none;
  padding: 5px 20px;
  background: rgba(10, 20, 50, 0.82);
  border-bottom: 1px solid rgba(121,134,203,0.25);
  font-family: 'Helvetica Neue', Arial, sans-serif;
  z-index: 99997;
}
#aal-badge .aal-region {
  color: #9fa8da;
  font-weight: 600;
  font-size: 13px;
  margin-right: 8px;
}
#aal-badge .aal-coords {
  color: #7986cb;
  font-size: 11px;
}
/* push viewer canvas below title bar */
#div_viewer { padding-top: 42px; }
</style>
"""

_TITLE_HTML = """
<div id="pet-title-bar">
  <span id="pet-title-main">{title}</span>
  <span id="pet-title-sub">{contrast_label}</span>
</div>
<div id="aal-badge">
  <span class="aal-region" id="aal-region-text"></span>
  <span class="aal-coords"  id="aal-coords-text"></span>
</div>
"""

_AAL_JS = """
<script>
(function(){{
  const STEP = {step};
  const LUT  = {lut_json};

  function snap(v)  {{ return Math.round(v / STEP) * STEP; }}
  function lookup(x, y, z) {{
    return LUT[snap(x) + "_" + snap(y) + "_" + snap(z)] || null;
  }}

  const badge  = document.getElementById("aal-badge");
  const region = document.getElementById("aal-region-text");
  const coords = document.getElementById("aal-coords-text");

  function showBadge(r, mx, my, mz) {{
    region.textContent = r ? r : "Outside atlas";
    coords.textContent = "(" + mx + ", " + my + ", " + mz + ")";
    badge.style.display = "block";
    clearTimeout(badge._t);
    badge._t = setTimeout(() => {{ badge.style.display = "none"; }}, 5000);
  }}

  function hookBrain() {{
    const b = window._brain;
    if (!b || !b.affine) {{ setTimeout(hookBrain, 100); return; }}
    b.onclick = function() {{
      const vx = b.numSlice.X, vy = b.numSlice.Y, vz = b.numSlice.Z;
      const A  = b.affine;
      const mx = Math.round(A[0][0]*vx + A[0][1]*vy + A[0][2]*vz + A[0][3]);
      const my = Math.round(A[1][0]*vx + A[1][1]*vy + A[1][2]*vz + A[1][3]);
      const mz = Math.round(A[2][0]*vx + A[2][1]*vy + A[2][2]*vz + A[2][3]);
      showBadge(lookup(mx, my, mz), mx, my, mz);
    }};
    console.log("AAL hook ready — " + Object.keys(LUT).length + " voxels indexed");
  }}

  setTimeout(hookBrain, 500);
}})();
</script>
"""


def show_brain_viewer(
    img,
    threshold: float,
    title: str,
    contrast_label: str,
    fname: str,
    out_dir: str | Path,
    lut_json: str,
    step: int = 2,
    bg_img=None,
    cmap: str = 'cold_hot',
    opacity: float = 0.8,
    height: int = 520,
) -> str:
    """
    Generate an interactive HTML brain viewer with AAL3 anatomical labeling.

    Produces two files in ``out_dir``:
    - ``<fname>``         — raw nilearn HTML viewer
    - ``<fname_aal.html>``— injected version with title bar + click badge

    The ``_aal.html`` file is fully self-contained (no server, no internet
    required) and is the file intended for sharing.

    Parameters
    ----------
    img : NIfTI image
        Statistical map to display (t-map, z-map, etc.).
    threshold : float
        Display threshold. Voxels with |value| < threshold are not rendered.
        Set to the t-value corresponding to your statistical threshold
        (e.g. t = 3.5 for p<0.001 with ~35 df).
    title : str
        Short contrast label shown prominently in the title bar
        (e.g. ``"C > B"``).
    contrast_label : str
        Statistical qualifier shown in smaller text in the title bar
        (e.g. ``"p<0.001 uncorrected (M1)"``).
    fname : str
        Filename for the raw nilearn output (e.g. ``"viewer_C_vs_B.html"``).
        The injected file is saved as ``fname`` with ``_aal`` inserted before
        the ``.html`` extension.
    out_dir : str or Path
        Directory where both HTML files are written.
    lut_json : str
        JSON string produced by ``lut_to_json()``. Built once per session
        and reused across all contrasts.
    step : int, optional
        Resolution of the LUT in mm. Must match ``step_mm`` used in
        ``build_aal_lookup()``. Default 2.
    bg_img : NIfTI image, optional
        Background anatomical image. Defaults to MNI152 template.
    cmap : str, optional
        Colormap for the statistical overlay. Default ``'cold_hot'``
        (blue–red, appropriate for signed t-maps). For one-sided maps
        consider ``'hot'`` or ``'inferno'``.
    opacity : float, optional
        Opacity of the statistical overlay, in [0, 1]. Default 0.8.
        Reduce to ~0.6 to show more anatomy through the overlay.
    height : int, optional
        Height in pixels of the IFrame rendered inline in Jupyter.
        Default 520. Does not affect the saved HTML file.

    Returns
    -------
    str
        Absolute path to the injected ``_aal.html`` file.
    """
    try:
        from IPython.display import IFrame, display as _display
        _in_jupyter = True
    except ImportError:
        _in_jupyter = False

    if bg_img is None:
        from nilearn import datasets as _ds
        bg_img = _ds.load_mni152_template()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Generate base nilearn viewer ──────────────────────────────────────
    v = plotting.view_img(
        img,
        bg_img=bg_img,
        threshold=threshold,
        cmap=cmap,
        symmetric_cmap=True,
        title='',        # title handled by injected HTML
        colorbar=True,
        opacity=opacity,
    )

    fpath = out_dir / fname
    v.save_as_html(str(fpath))

    with open(fpath, 'r', encoding='utf-8') as f:
        html = f.read()

    # ── Expose brainsprite instance as window._brain ──────────────────────
    html = html.replace(
        'var brain = brainsprite(',
        'var brain = window._brain = brainsprite('
    )

    # ── Build injection block ─────────────────────────────────────────────
    title_block = _TITLE_HTML.format(
        title=title,
        contrast_label=contrast_label,
    )
    js_block = _AAL_JS.format(step=step, lut_json=lut_json)
    injection = _BADGE_CSS + title_block + js_block

    if '</body>' in html:
        html_inj = html.replace('</body>', injection + '\n</body>', 1)
    else:
        html_inj = html + injection

    # ── Save injected file ────────────────────────────────────────────────
    fpath_inj = out_dir / fname.replace('.html', '_aal.html')
    with open(fpath_inj, 'w', encoding='utf-8') as f:
        f.write(html_inj)

    print(f"  Saved : {fpath_inj.name}")

    if _in_jupyter:
        _display(IFrame(src=str(fpath_inj), width='100%', height=f'{height}px'))

    return str(fpath_inj)
