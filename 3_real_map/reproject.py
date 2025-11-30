#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import array_bounds


def tighten_and_reproject_to_5070(input_tif, output_tif, use_nodata=True):
    """
    1) Tight-crop a clipped CONUS raster to the minimal non-empty window.
    2) Reproject that tight subset to EPSG:5070 (NAD83 / Conus Albers).

    Parameters
    ----------
    input_tif : str or Path
        Path to *clipped* CONUS raster (e.g. nightlights_conus_clipped.tif).
    output_tif : str or Path
        Path to write the reprojected, tight raster.
    use_nodata : bool
        If True, treat src.nodata as "empty" when tightening.
        If False or nodata is None, treat value <= 0 as empty.
    """

    input_tif = Path(input_tif)
    output_tif = Path(output_tif)
    dst_crs = "EPSG:5070"

    with rasterio.open(input_tif) as src:
        print("Source CRS:", src.crs)
        print("Target CRS:", dst_crs)
        print("Source size:", src.width, "x", src.height)
        print("Source bounds:", src.bounds)

        # --- Read full band once, then find tight window ---
        data = src.read(1)

        if use_nodata and src.nodata is not None:
            nodata_val = src.nodata
            mask = data != nodata_val
        else:
            # treat <= 0 as "empty" (common for masked night lights)
            mask = data > 0

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]

        if rows.size == 0 or cols.size == 0:
            raise RuntimeError("No non-empty pixels found to tighten around.")

        r0, r1 = rows[0], rows[-1]
        c0, c1 = cols[0], cols[-1]

        print(f"Tight window (row {r0}:{r1+1}, col {c0}:{c1+1})")

        window = Window.from_slices((r0, r1 + 1), (c0, c1 + 1))
        src_tight = src.read(1, window=window)
        src_tight_transform = rasterio.windows.transform(window, src.transform)

        H, W = src_tight.shape
        print("Tight size:", W, "x", H)

        # --- Compute bounds of the tight subset in source CRS ---
        left, bottom, right, top = array_bounds(H, W, src_tight_transform)

        # --- Calculate target transform & shape in EPSG:5070 ---
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, W, H, left, bottom, right, top
        )

        print("Reprojected size:", width, "x", height)

        # Prepare output metadata
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height
        })

        # Allocate destination array
        dst_array = np.empty((height, width), dtype=src.dtypes[0])

        # --- Reproject just the tight subset ---
        print("Reprojecting (this may take a while)...")
        reproject(
            source=src_tight,
            destination=dst_array,
            src_transform=src_tight_transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )

    # --- Write result ---
    print("Writing:", output_tif)
    output_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_tif, "w", **kwargs) as dst:
        dst.write(dst_array, 1)

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python tight_reproject_conus.py <input_clipped.tif> <output_5070_tight.tif>")
        sys.exit(1)

    in_tif = sys.argv[1]
    out_tif = sys.argv[2]
    tighten_and_reproject_to_5070(in_tif, out_tif)
