"""
gee.py — Earth Engine initialisation and live preview tile layers.

This module is self-contained (depends only on `earthengine-api`); it does NOT
import your processing Modules. It powers the interactive map preview in the
toolkit: fast getMapId() XYZ tiles for an MNDWI water composite and the JRC
Global Surface Water occurrence layer, without exporting anything.

Auth: either user credentials (run `earthengine authenticate` once) or a
service account (set EE_SERVICE_ACCOUNT and EE_KEY_FILE). Set EE_PROJECT to your
Cloud project id.
"""
from __future__ import annotations
import os

_INITIALISED = False


def init_ee() -> None:
    """Initialise Earth Engine once. Safe to call repeatedly."""
    global _INITIALISED
    if _INITIALISED:
        return
    import ee
    project = os.getenv("EE_PROJECT") or None
    sa = os.getenv("EE_SERVICE_ACCOUNT")
    key = os.getenv("EE_KEY_FILE")
    if sa and key:
        creds = ee.ServiceAccountCredentials(sa, key)
        ee.Initialize(creds, project=project)
    else:
        # Falls back to cached user credentials from `earthengine authenticate`.
        ee.Initialize(project=project)
    _INITIALISED = True


# ---- Landsat surface-reflectance helpers (Collection 2, Level 2) ----------

_BANDS = {
    "LANDSAT/LT05/C02/T1_L2": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
    "LANDSAT/LE07/C02/T1_L2": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
    "LANDSAT/LC08/C02/T1_L2": ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
    "LANDSAT/LC09/C02/T1_L2": ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
}
_COMMON = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]


def _prep(cid):
    import ee

    def _f(img):
        img = img.select(_BANDS[cid], _COMMON)
        opt = img.select(_COMMON).multiply(0.0000275).add(-0.2)
        qa = img.select("QA_PIXEL") if False else None  # QA carried below
        return opt.copyProperties(img, img.propertyNames())
    return _f


def _mask_clouds(img):
    import ee
    qa = img.select("QA_PIXEL")
    # bits 3 (cloud) and 4 (cloud shadow)
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    return img.updateMask(mask)


def _s2_collection(geom, start, end, max_cloud):
    """Sentinel-2 SR (harmonised) → Green/SWIR1 surface reflectance, cloud-masked via SCL."""
    import ee

    def _mask(img):
        scl = img.select("SCL")
        # drop saturated(1), shadow(3), cloud med/high(8,9), cirrus(10), snow(11)
        bad = (scl.eq(1).Or(scl.eq(3)).Or(scl.eq(8)).Or(scl.eq(9))
               .Or(scl.eq(10)).Or(scl.eq(11)))
        green = img.select("B3").multiply(1e-4).rename("Green")
        swir1 = img.select("B11").multiply(1e-4).rename("SWIR1")
        nir = img.select("B8").multiply(1e-4).rename("NIR")
        return (green.addBands(swir1).addBands(nir)
                .updateMask(bad.Not()).copyProperties(img, img.propertyNames()))

    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geom).filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
            .map(_mask))


def mndwi_water_composite(bbox, y0, y1, max_cloud=20, sensors=("landsat", "s2")):
    """Median MNDWI composite + binary water mask over the bbox / year range.

    Sentinel-2 (10–20 m) and/or Landsat (30 m) optical, per `sensors`.
    """
    import ee
    geom = ee.Geometry.Rectangle([bbox["w"], bbox["s"], bbox["e"], bbox["n"]])
    start, end = f"{int(y0)}-01-01", f"{int(y1)}-12-31"
    parts = []
    if "landsat" in sensors:
        for cid in _BANDS:
            c = (ee.ImageCollection(cid)
                 .filterBounds(geom).filterDate(start, end)
                 .filter(ee.Filter.lt("CLOUD_COVER", max_cloud))
                 .map(_mask_clouds))
            c = c.map(lambda im, cid=cid: im.select(_BANDS[cid], _COMMON)
                      .multiply(0.0000275).add(-0.2)
                      .copyProperties(im, im.propertyNames()))
            parts.append(c.select(["Green", "SWIR1"]))
    if "s2" in sensors:
        parts.append(_s2_collection(geom, start, end, max_cloud).select(["Green", "SWIR1"]))
    if not parts:
        raise ValueError("No optical sensor selected (landsat and/or s2).")
    merged = parts[0]
    for c in parts[1:]:
        merged = merged.merge(c)
    comp = merged.median().clip(geom)
    mndwi = comp.normalizedDifference(["Green", "SWIR1"]).rename("MNDWI")
    water = mndwi.gt(0).selfMask().rename("water")
    return geom, mndwi, water


def preview_layers(bbox, y0, y1, sensors=("landsat",), max_cloud=20):
    """Return XYZ tile templates for the map: MNDWI, water mask, JRC occurrence."""
    import ee
    init_ee()
    geom, mndwi, water = mndwi_water_composite(bbox, y0, y1, max_cloud, sensors=tuple(sensors))
    out = []

    def _tiles(img, vis, name):
        m = img.getMapId(vis)
        return {"name": name,
                "tile_url": m["tile_fetcher"].url_format,
                "attribution": "Google Earth Engine"}

    out.append(_tiles(mndwi, {"min": -0.6, "max": 0.6,
                              "palette": ["#c2ae92", "#f5f0e6", "#8fa2ab", "#5f7b86"]},
                      "MNDWI (median)"))
    out.append(_tiles(water, {"palette": ["#5f7b86"]}, "Water mask (MNDWI>0)"))
    jrc = (ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
           .select("occurrence").clip(geom))
    out.append(_tiles(jrc, {"min": 0, "max": 100,
                            "palette": ["#f5f0e6", "#9cab8a", "#5f7b86"]},
                      "JRC water occurrence"))
    return out


# ---- Sentinel-2 MNDWI export (feeds the existing Singularity→Otsu→Refiner chain) --

def export_s2_mndwi(bbox, y0, y1, output_dir, max_cloud=20, scale=10):
    """
    Download per-scene Sentinel-2 MNDWI GeoTIFFs named ``mndwi_YYYY-MM-DD.tif``
    into ``output_dir`` — the exact filename convention
    ``SingularityIndexProcessor(mndwi_pattern='mndwi_*.tif')`` consumes.

    This is the Sentinel-2 analogue of the Landsat ``MNDWIExporter``: it lets the
    whole downstream pipeline run on S2 (10 m) without touching your Modules.
    Requires `geemap`.
    """
    import os
    import ee
    import geemap
    init_ee()
    os.makedirs(output_dir, exist_ok=True)
    geom = ee.Geometry.Rectangle([bbox["w"], bbox["s"], bbox["e"], bbox["n"]])
    col = _s2_collection(geom, f"{int(y0)}-01-01", f"{int(y1)}-12-31", max_cloud)
    col = col.map(lambda im: im.normalizedDifference(["Green", "SWIR1"])
                  .rename("MNDWI").copyProperties(im, ["system:time_start"]))
    ids = col.aggregate_array("system:index").getInfo()
    imgs = col.toList(col.size())
    paths = []
    for i, _id in enumerate(ids):
        img = ee.Image(imgs.get(i)).clip(geom)
        dt = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd").getInfo()
        out = os.path.join(output_dir, f"mndwi_{dt}.tif")
        geemap.ee_export_image(img, filename=out, scale=scale, region=geom,
                               crs="EPSG:32646", file_per_band=False)
        paths.append(out)
    return paths
