"""
gbm_delta.viz.map_gif
=====================
Annual map animations and subplot figures from intertidal-elevation
GeoTIFF stacks — cleaned from the ``export_map_overlay`` notebook.

Conventions (same as the notebook):
    raster value 999  → land  (light grey)
    NaN              → river (white)
    other values     → intertidal elevation, coloured with a truncated
                       cmocean (or matplotlib) colormap

Two bugs from the notebook are fixed here:
    * ``draw`` was undefined when a filename carried no 4-digit year
      (the scale bar crashed);
    * non-cmocean colormap names fell through to an undefined
      ``colormaps`` mapping — now any matplotlib colormap name works,
      and cmocean is optional.

Requires: pillow (+ optional cmocean, seaborn).
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont

try:
    import cmocean
    _HAS_CMO = True
except ImportError:
    _HAS_CMO = False


# ── colormaps ────────────────────────────────────────────────────────────────
def truncate_colormap(cmap, minval: float = 0.2, maxval: float = 1.0,
                      n: int = 256):
    """Truncate a colormap to the [minval, maxval] sub-range."""
    from matplotlib.colors import LinearSegmentedColormap
    new_colors = cmap(np.linspace(minval, maxval, n))
    return LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})", new_colors)


def _resolve_cmap(colormap: str = "cmo.deep"):
    """Resolve 'cmo.<name>' (cmocean, optional) or any matplotlib name."""
    import matplotlib.pyplot as plt
    if colormap.startswith("cmo."):
        if not _HAS_CMO:
            print("cmocean not installed — falling back to 'viridis'. "
                  "pip install cmocean for the original look.")
            return truncate_colormap(plt.get_cmap("viridis"))
        return truncate_colormap(getattr(cmocean.cm, colormap.split(".")[-1]))
    return truncate_colormap(plt.get_cmap(colormap))


def _load_font(font_size: int = 20):
    """Arial → Liberation Sans → PIL default bitmap font."""
    for name in ("arial.ttf", "LiberationSans-Regular.ttf",
                 "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, font_size)
        except OSError:
            continue
    print("Fallback to default font. Consider installing Arial.")
    return ImageFont.load_default()


# ── colorbar / scale bar ─────────────────────────────────────────────────────
def generate_colorbar(vmin, vmax, colormap="cmo.deep", height=800, width=80,
                      label="Elevation"):
    """Vertical colorbar image (PIL) matching the map's vmin/vmax."""
    import io
    import matplotlib.pyplot as plt

    dpi = 200
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(top=0.90, bottom=0.10)

    cmap = colormap if hasattr(colormap, "__call__") else _resolve_cmap(colormap)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=ax, orientation="vertical")
    cbar.set_label(label, fontsize=10, color="black", labelpad=10)
    cbar.ax.tick_params(labelsize=8, colors="black")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def draw_scale_bar(draw, image_width, image_height,
                   scale_length_px=100, scale_text="2 km"):
    """Draw a simple scale bar at the bottom-left of a PIL image."""
    margin = 10
    bar_height = 8
    bar_top = image_height - margin - bar_height
    bar_left = margin
    bar_right = bar_left + scale_length_px
    draw.rectangle([bar_left, bar_top, bar_right, bar_top + bar_height],
                   fill="black")
    draw.text((bar_left, bar_top - 20), scale_text, fill="black")


# ── GIF ──────────────────────────────────────────────────────────────────────
def create_tif_gif_bar(input_folder, output_gif, output_folder=None,
                       duration=500, resize_ratio=1.0, colormap="cmo.deep",
                       font_size=20, mask_color=(255, 255, 255)):
    """
    GIF animation of all GeoTIFFs in `input_folder`, with year annotation
    and a fixed (global-percentile) colorbar.
    """
    tif_files = sorted(f for f in os.listdir(input_folder) if f.endswith(".tif"))
    if not tif_files:
        raise ValueError("No TIFF files found in the specified folder")
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # global 2–98 % range over all valid pixels
    all_valid = []
    for tif_file in tif_files:
        with rasterio.open(os.path.join(input_folder, tif_file)) as src:
            data = src.read(1)
            v = data[(data != 999) & ~np.isnan(data)]
            if v.size:
                all_valid.extend(v.ravel())
    vmin, vmax = np.nanpercentile(all_valid, [2, 98])

    font = _load_font(font_size)
    cmap = _resolve_cmap(colormap)
    images = []

    for tif_file in tif_files:
        # year from the trailing _YYYY token (may be absent → None)
        try:
            year = tif_file.split("_")[-1].split(".")[0]
            if not year.isdigit() or len(year) != 4:
                year = None
        except Exception:
            year = None

        with rasterio.open(os.path.join(input_folder, tif_file)) as src:
            data = src.read(1)

        land_mask = (data == 999)
        water_mask = np.isnan(data)
        valid_mask = ~(land_mask | water_mask)

        norm_data = np.clip((data - vmin) / (vmax - vmin), 0, 1)
        rgb = np.ones((*data.shape, 3), dtype=np.uint8) * 255      # river
        colored = (cmap(norm_data)[:, :, :3] * 255).astype(np.uint8)
        rgb[valid_mask] = colored[valid_mask]
        rgb[land_mask] = [200, 200, 200]                            # land

        img = Image.fromarray(rgb)
        if resize_ratio != 1.0:
            img = img.resize((int(img.width * resize_ratio),
                              int(img.height * resize_ratio)), Image.LANCZOS)

        colorbar_img = generate_colorbar(vmin, vmax, colormap=cmap,
                                         height=img.height)
        combined = Image.new("RGB",
                             (img.width + colorbar_img.width, img.height),
                             (255, 255, 255))
        combined.paste(img, (0, 0))
        combined.paste(colorbar_img.convert("RGB"), (img.width, 0))

        # FIX: always create the draw object, even when no year was parsed
        draw = ImageDraw.Draw(combined)
        if year:
            for x_offset in (-1, 0, 1):
                for y_offset in (-1, 0, 1):
                    if x_offset or y_offset:
                        draw.text((10 + x_offset, 10 + y_offset), year,
                                  fill="white", font=font)
            draw.text((10, 10), year, fill="black", font=font)

        draw_scale_bar(draw, combined.width, combined.height,
                       scale_length_px=100, scale_text="2 km")

        if output_folder:
            combined.save(os.path.join(
                output_folder, f"IA_map_{year if year else tif_file}.png"))
        images.append(combined)

    if images:
        images[0].save(output_gif, save_all=True,
                       append_images=images[1:], duration=duration,
                       loop=0, optimize=True)
        print(f"GIF created successfully: {output_gif}")
    else:
        print("No valid images were processed.")


# ── subplots ─────────────────────────────────────────────────────────────────
def make_subplot(image_folder, selected_years, output_path=None):
    """2×3 subplot of previously exported ``IA_map_YYYY.png`` panels."""
    import matplotlib.pyplot as plt
    n = len(selected_years)
    if n != 6:
        raise ValueError("Exactly 6 years must be selected for a 2x3 subplot.")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for ax, year in zip(axes, selected_years):
        img = Image.open(os.path.join(image_folder, f"IA_map_{year}.png"))
        ax.imshow(img)
        ax.set_title(str(year), fontsize=14)
        ax.axis("off")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Subplot saved to {output_path}")
    else:
        plt.show()


def _draw_scale_bar_axes(ax, length_px=100, text="2 km",
                         location=(0.05, 0.95), bar_height=5,
                         color="black", fontsize=10):
    """Scale bar on a matplotlib axes at a relative position."""
    from matplotlib.patches import Rectangle
    x0, y0 = location
    ax_coords = ax.transAxes.transform((x0, y0))
    inv = ax.transData.inverted()
    x_data, y_data = inv.transform(ax_coords)

    ax.add_patch(Rectangle((x_data, y_data - bar_height), length_px,
                           bar_height, facecolor=color, edgecolor="none"))
    px_off = (inv.transform((0, 1))[1] - inv.transform((0, 0))[1])
    ax.text(x_data + length_px / 2, y_data - bar_height - px_off,
            text, ha="center", va="top", color=color, fontsize=fontsize)


def create_subplot_figure(input_folder, selected_years, output_path,
                          colormap="cmo.deep", font_size=14,
                          figsize=(12, 8)):
    """
    Static 2×3 figure for six selected years with one shared colorbar.
    Assumes ~pixel-size units so the scale bar is correct at 30 m pixels
    unless `length_px` is adapted.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    tif_files = {f.split("_")[-1].split(".")[0]: f
                 for f in os.listdir(input_folder) if f.endswith(".tif")}
    selected_files = [tif_files[str(y)] for y in selected_years
                      if str(y) in tif_files]
    if len(selected_files) != 6:
        raise ValueError("Exactly 6 matching years must be selected "
                         "for subplot visualization.")

    all_valid = []
    res_x = 30.0
    for fname in selected_files:
        with rasterio.open(os.path.join(input_folder, fname)) as src:
            arr = src.read(1)
            res_x = src.res[0]
            all_valid.extend(arr[(arr != 999) & ~np.isnan(arr)])
    vmin, vmax = np.percentile(all_valid, [2, 98])
    scale_length_px = int(20000 / res_x)          # 20 km at native resolution

    cmap = _resolve_cmap(colormap)
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=False)
    fig.subplots_adjust(right=0.9)
    axes = axes.flatten()

    for ax, fname, year in zip(axes, selected_files, selected_years):
        with rasterio.open(os.path.join(input_folder, fname)) as src:
            arr = src.read(1)
        land_mask = (arr == 999)
        water_mask = np.isnan(arr)
        valid_mask = ~(land_mask | water_mask)
        rgba = np.ones((*arr.shape, 3))
        rgba[valid_mask] = cmap(norm(arr))[:, :, :3][valid_mask]
        rgba[land_mask] = [0.8, 0.8, 0.8]
        ax.imshow(rgba)
        ax.set_title(str(year), fontsize=font_size)
        ax.axis("off")
        _draw_scale_bar_axes(ax, length_px=scale_length_px, text="20 km",
                            fontsize=5)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), cax=cbar_ax)
    cbar.ax.tick_params(labelsize=font_size)
    cbar.set_label("Elevation", fontsize=font_size)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved subplot figure to: {output_path}")
