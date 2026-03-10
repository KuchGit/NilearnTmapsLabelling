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

  function mat4inv(m) {{
    // m is b.affine: array of 4 rows, each row is array of 4
    // Flatten to 16-element array, compute inverse, return same structure
    var a = [
      m[0][0],m[0][1],m[0][2],m[0][3],
      m[1][0],m[1][1],m[1][2],m[1][3],
      m[2][0],m[2][1],m[2][2],m[2][3],
      m[3][0],m[3][1],m[3][2],m[3][3]
    ];
    var inv = new Array(16);
    inv[0]  =  a[5]*a[10]*a[15]-a[5]*a[11]*a[14]-a[9]*a[6]*a[15]+a[9]*a[7]*a[14]+a[13]*a[6]*a[11]-a[13]*a[7]*a[10];
    inv[4]  = -a[4]*a[10]*a[15]+a[4]*a[11]*a[14]+a[8]*a[6]*a[15]-a[8]*a[7]*a[14]-a[12]*a[6]*a[11]+a[12]*a[7]*a[10];
    inv[8]  =  a[4]*a[9]*a[15]-a[4]*a[11]*a[13]-a[8]*a[5]*a[15]+a[8]*a[7]*a[13]+a[12]*a[5]*a[11]-a[12]*a[7]*a[9];
    inv[12] = -a[4]*a[9]*a[14]+a[4]*a[10]*a[13]+a[8]*a[5]*a[14]-a[8]*a[6]*a[13]-a[12]*a[5]*a[10]+a[12]*a[6]*a[9];
    inv[1]  = -a[1]*a[10]*a[15]+a[1]*a[11]*a[14]+a[9]*a[2]*a[15]-a[9]*a[3]*a[14]-a[13]*a[2]*a[11]+a[13]*a[3]*a[10];
    inv[5]  =  a[0]*a[10]*a[15]-a[0]*a[11]*a[14]-a[8]*a[2]*a[15]+a[8]*a[3]*a[14]+a[12]*a[2]*a[11]-a[12]*a[3]*a[10];
    inv[9]  = -a[0]*a[9]*a[15]+a[0]*a[11]*a[13]+a[8]*a[1]*a[15]-a[8]*a[3]*a[13]-a[12]*a[1]*a[11]+a[12]*a[3]*a[9];
    inv[13] =  a[0]*a[9]*a[14]-a[0]*a[10]*a[13]-a[8]*a[1]*a[14]+a[8]*a[2]*a[13]+a[12]*a[1]*a[10]-a[12]*a[2]*a[9];
    inv[2]  =  a[1]*a[6]*a[15]-a[1]*a[7]*a[14]-a[5]*a[2]*a[15]+a[5]*a[3]*a[14]+a[13]*a[2]*a[7]-a[13]*a[3]*a[6];
    inv[6]  = -a[0]*a[6]*a[15]+a[0]*a[7]*a[14]+a[4]*a[2]*a[15]-a[4]*a[3]*a[14]-a[12]*a[2]*a[7]+a[12]*a[3]*a[6];
    inv[10] =  a[0]*a[5]*a[15]-a[0]*a[7]*a[13]-a[4]*a[1]*a[15]+a[4]*a[3]*a[13]+a[12]*a[1]*a[7]-a[12]*a[3]*a[5];
    inv[14] = -a[0]*a[5]*a[14]+a[0]*a[6]*a[13]+a[4]*a[1]*a[14]-a[4]*a[2]*a[13]-a[12]*a[1]*a[6]+a[12]*a[2]*a[5];
    inv[3]  = -a[1]*a[6]*a[11]+a[1]*a[7]*a[10]+a[5]*a[2]*a[11]-a[5]*a[3]*a[10]-a[9]*a[2]*a[7]+a[9]*a[3]*a[6];
    inv[7]  =  a[0]*a[6]*a[11]-a[0]*a[7]*a[10]-a[4]*a[2]*a[11]+a[4]*a[3]*a[10]+a[8]*a[2]*a[7]-a[8]*a[3]*a[6];
    inv[11] = -a[0]*a[5]*a[11]+a[0]*a[7]*a[9]+a[4]*a[1]*a[11]-a[4]*a[3]*a[9]-a[8]*a[1]*a[7]+a[8]*a[3]*a[5];
    inv[15] =  a[0]*a[5]*a[10]-a[0]*a[6]*a[9]-a[4]*a[1]*a[10]+a[4]*a[2]*a[9]+a[8]*a[1]*a[6]-a[8]*a[2]*a[5];
    var det = a[0]*inv[0]+a[1]*inv[4]+a[2]*inv[8]+a[3]*inv[12];
    if (det === 0) return null;
    for (var i=0; i<16; i++) inv[i] /= det;
    return inv;
  }}

  function navigateTo(mx, my, mz) {{
    var b = window._brain;
    if (!b || !b.affine) {{ return; }}
    var inv = mat4inv(b.affine);
    if (!inv) return;
    var vx = inv[0]*mx + inv[1]*my + inv[2]*mz  + inv[3];
    var vy = inv[4]*mx + inv[5]*my + inv[6]*mz  + inv[7];
    var vz = inv[8]*mx + inv[9]*my + inv[10]*mz + inv[11];
    b.numSlice.X = vx; b.numSlice.Y = vy; b.numSlice.Z = vz;
    b.draw(vx, "X");
    b.draw(vy, "Y");
    b.draw(vz, "Z");
    showBadge(lookup(mx, my, mz), mx, my, mz);
  }}
  window.navigateTo = navigateTo;

  setTimeout(hookBrain, 500);
}})();
</script>
"""


_TABLE_CSS = """
<style>
#pet-cluster-table {
  font-family: 'Helvetica Neue', Arial, sans-serif;
  font-size: 12px;
  padding: 12px 20px 20px 20px;
  background: #ffffff;
  color: #24292f;
}
#pet-cluster-table h4 {
  font-size: 13px;
  font-weight: 600;
  margin: 0 0 8px 0;
  color: #24292f;
}
#pet-cluster-table table {
  border-collapse: collapse;
  width: auto;
}
#pet-cluster-table thead th {
  background: #f6f8fa;
  border: 1px solid #d0d7de;
  padding: 5px 12px;
  text-align: left;
  font-weight: 600;
  font-size: 11px;
  color: #57606a;
  letter-spacing: 0.04em;
  white-space: nowrap;
  cursor: pointer;
  user-select: none;
}
#pet-cluster-table thead th:hover { background: #e8f0fe; color: #1a56db; }
#pet-cluster-table thead th.sort-asc::after  { content: " ▲"; font-size: 9px; }
#pet-cluster-table thead th.sort-desc::after { content: " ▼"; font-size: 9px; }
#pet-cluster-table tbody td {
  border: 1px solid #d0d7de;
  padding: 4px 12px;
  white-space: nowrap;
}
#pet-cluster-table tbody tr { cursor: pointer; }
#pet-cluster-table tbody tr.active td { background: #fff3cd !important; font-weight: 600; }
</style>
<script>
(function() {
  document.addEventListener("DOMContentLoaded", function() {
    var tables = document.querySelectorAll("#pet-cluster-table table");
    tables.forEach(function(table) {
      // ── sort ──
      var headers = table.querySelectorAll("thead th");
      var sortState = { col: -1, asc: true };
      headers.forEach(function(th, colIdx) {
        th.addEventListener("click", function() {
          var asc = (sortState.col === colIdx) ? !sortState.asc : true;
          sortState = { col: colIdx, asc: asc };
          headers.forEach(function(h) { h.className = ""; });
          th.className = asc ? "sort-asc" : "sort-desc";
          var tbody = table.querySelector("tbody");
          var rows  = Array.from(tbody.querySelectorAll("tr"));
          rows.sort(function(a, b) {
            var va = a.cells[colIdx].textContent.trim();
            var vb = b.cells[colIdx].textContent.trim();
            var na = parseFloat(va.replace(/,/g, ""));
            var nb = parseFloat(vb.replace(/,/g, ""));
            if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
            return asc ? va.localeCompare(vb) : vb.localeCompare(va);
          });
          rows.forEach(function(r) { tbody.appendChild(r); });
        });
      });
      // ── row click → navigate ──
      table.querySelector("tbody").addEventListener("click", function(e) {
        var tr = e.target.closest("tr");
        if (!tr) return;
        var x = parseFloat(tr.dataset.x);
        var y = parseFloat(tr.dataset.y);
        var z = parseFloat(tr.dataset.z);
        if (isNaN(x)) return;
        // highlight
        table.querySelectorAll("tbody tr").forEach(function(r) { r.classList.remove("active"); });
        tr.classList.add("active");
        if (typeof window.navigateTo === "function") window.navigateTo(x, y, z);
      });
    });
  });
})();
</script>
"""


def _df_to_html(df, title, lut=None, step=2):
    """Convert a clusters DataFrame to styled HTML with optional AAL ROI column."""
    import pandas as pd

    df = df.copy()

    # Add ROI column from AAL lookup on peak coordinates
    if lut is not None and {'X', 'Y', 'Z'}.issubset(df.columns):
        def _lookup(row):
            xs = int(round(row['X'] / step) * step)
            ys = int(round(row['Y'] / step) * step)
            zs = int(round(row['Z'] / step) * step)
            return lut.get('%d_%d_%d' % (xs, ys, zs), '—')
        df.insert(0, 'ROI (AAL3)', df.apply(_lookup, axis=1))

    # Header
    thead = '<tr>' + ''.join(f'<th>{c}</th>' for c in df.columns) + '</tr>'

    # Rows — encode MNI coords as data attributes for JS navigation
    rows = ''
    has_xyz = {'X', 'Y', 'Z'}.issubset(df.columns)
    for _, row in df.iterrows():
        data_attr = ''
        if has_xyz:
            data_attr = f' data-x="{row["X"]}" data-y="{row["Y"]}" data-z="{row["Z"]}"'
        cells = ''
        for v in row:
            if isinstance(v, float):
                cells += f'<td>{v:.2f}</td>'
            else:
                cells += f'<td>{v}</td>'
        rows += f'<tr{data_attr}>{cells}</tr>'

    return (
        f'<div id="pet-cluster-table">'
        f'<h4>{title}</h4>'
        f'<table><thead>{thead}</thead><tbody>{rows}</tbody></table>'
        f'</div>'
    )


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
    clusters_table=None,
    clusters_table_title: str = 'Cluster table',
    lut: dict = None,
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

    # ── Append cluster table below viewer ────────────────────────────────
    if clusters_table is not None:
        table_html = _TABLE_CSS + _df_to_html(clusters_table, clusters_table_title, lut=lut, step=step)
        if '</html>' in html_inj:
            html_inj = html_inj.replace('</html>', table_html + '\n</html>', 1)
        else:
            html_inj += table_html

    # ── Save injected file ────────────────────────────────────────────────
    fpath_inj = out_dir / fname.replace('.html', '_aal.html')
    with open(fpath_inj, 'w', encoding='utf-8') as f:
        f.write(html_inj)

    print(f"  Saved : {fpath_inj.name}")

    if _in_jupyter:
        _display(IFrame(src=str(fpath_inj), width='100%', height=f'{height}px'))

    return str(fpath_inj)