#!/usr/bin/env python

import sys
from pathlib import Path

import rasterio
from rasterio.mask import mask
import geopandas as gpd


def clip_conus(
    tif_path,
    shp_path,
    out_path,
    shapefile_layer=None,
    shapefile_filter_column=None,
    shapefile_filter_values=None,
):
    """
    Clip a big raster to the continental US using a polygon shapefile.

    Parameters
    ----------
    tif_path : str or Path
        Path to input GeoTIFF (e.g. night lights tile).
    shp_path : str or Path
        Path to a US boundary shapefile / geopackage.
    out_path : str or Path
        Path to output clipped GeoTIFF.
    shapefile_layer : str, optional
        Layer name if shp_path is a GeoPackage, etc.
    shapefile_filter_column : str, optional
        Column name to filter (e.g. 'STATEFP' or 'NAME').
    shapefile_filter_values : list, optional
        Values in that column to keep (e.g. ['CA','TX',...] or ['Alabama',...]).
        Use this if your shapefile includes Alaska/Hawaii and you only want CONUS.
    """

    tif_path = Path(tif_path)
    shp_path = Path(shp_path)
    out_path = Path(out_path)

    print(f"Reading raster: {tif_path}")
    with rasterio.open(tif_path) as src:
        raster_crs = src.crs
        raster_meta = src.meta.copy()
        print(f"  CRS: {raster_crs}")
        print(f"  Size: {src.width} x {src.height}")
        print(f"  Bounds: {src.bounds}")

        # --- Load shapefile / geopackage ---
        print(f"Reading shapefile: {shp_path}")
        if shapefile_layer is not None:
            us = gpd.read_file(shp_path, layer=shapefile_layer)
        else:
            us = gpd.read_file(shp_path)

        print(f"  Shapefile CRS: {us.crs}")
        print(f"  Features: {len(us)}")

        # Optional attribute filter to get only CONUS states
        if shapefile_filter_column is not None and shapefile_filter_values is not None:
            before = len(us)
            us = us[us[shapefile_filter_column].isin(shapefile_filter_values)]
            print(f"  Filtered features: {len(us)} (from {before})")

        # Reproject polygon to raster CRS (cheap, no raster distortion)
        if us.crs != raster_crs:
            print("Reprojecting shapefile to raster CRS...")
            us = us.to_crs(raster_crs)

        # Union all geometries into one CONUS polygon
        us_geom = [us.unary_union.__geo_interface__]

        # --- Clip (mask) the raster ---
        print("Clipping raster with US polygon (this may take a while)...")
        out_image, out_transform = mask(
            src,
            us_geom,
            crop=True,
            nodata=raster_meta.get("nodata", 0)  # outside polygon
        )

        # Update metadata for output
        out_meta = raster_meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

    # --- Write result to disk ---
    print(f"Writing clipped raster to: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(out_image)

    print("Done.")
    print(f"  Output size: {out_image.shape[2]} x {out_image.shape[1]}")
    print(f"  Bands: {out_image.shape[0]}")
    print("  You now have a CONUS-only GeoTIFF.")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Usage: python clip_conus.py <input.tif> <us_shapefile> <output.tif>\n"
            "Example: python clip_conus.py nightlights.tif us_conus.shp nightlights_conus.tif"
        )
        sys.exit(1)

    tif = sys.argv[1]
    shp = sys.argv[2]
    out = sys.argv[3]

    # --- Minimal default: assume shapefile already is CONUS-only ---
    # If your shapefile has all US states (including AK/HI) and you want
    # to filter it, uncomment & adapt the example below.

    clip_conus(
        tif_path=tif,
        shp_path=shp,
        out_path=out,

        # Example: if using a Census TIGER states shapefile with 'STUSPS'
        # as the 2-letter state code column:
        #
        # shapefile_filter_column="STUSPS",
        # shapefile_filter_values=[
        #     "AL","AZ","AR","CA","CO","CT","DE","FL","GA",
        #     "IA","ID","IL","IN","KS","KY","LA","MA","MD","ME",
        #     "MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ",
        #     "NM","NV","NY","OH","OK","OR","PA","RI","SC","SD",
        #     "TN","TX","UT","VA","VT","WA","WI","WV","WY"
        # ],
    )