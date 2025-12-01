#!/usr/bin/env python3

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import rasterio
from rasterio.transform import xy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def build_clusters(
    tif_path,
    x_spacing_km,
    y_spacing_km,
    clustering_number,
    out_prefix=None,
):
    """
    From a reprojected (EPSG:5070), tightly clipped CONUS raster, do:

    1) Aggregate into a grid with approximate spacing (x_spacing_km, y_spacing_km).
    2) Treat each non-zero grid cell as a sample point with a weight
       equal to its summed demand.
    3) Run weighted KMeans to get `clustering_number` clusters.
    4) Use a local coordinate system where (0,0) is at the center of the grid:
         x_graph_km = x_real_km - x0_km
         y_graph_km = y_real_km - y0_km
       (x0_km, y0_km are printed & saved.)

    Outputs are all tagged with a run ID:
      - <out_prefix>_ID-<run_id>_clusters.csv
      - <out_prefix>_ID-<run_id>_grid_points.csv
      - <out_prefix>_ID-<run_id>_demand_grid.png
      - <out_prefix>_ID-<run_id>_cluster_centers.png
      - <out_prefix>_ID-<run_id>_coord_shift.txt
    """

    tif_path = Path(tif_path)
    if out_prefix is None:
        out_prefix = tif_path.with_suffix("")
    else:
        out_prefix = Path(out_prefix)

    # --- Run ID (for all outputs) ---
    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    print(f"Run ID: {run_id}")
    tagged_prefix = f"{out_prefix}_ID-{run_id}"

    # --- Create output directory ---
    out_dir = Path(tagged_prefix)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to directory: {out_dir}")

    # --- Load raster ---
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        transform = src.transform
        nodata = src.nodata
        crs = src.crs
        H, W = data.shape

        pixel_w_m = transform.a
        pixel_h_m = abs(transform.e)

    print(f"Loaded {tif_path}")
    print(f"  CRS: {crs}")
    print(f"  Size: {W} x {H}")
    print(f"  Pixel size: {pixel_w_m:.3f} m (x), {pixel_h_m:.3f} m (y)")

    # --- Determine grid dimensions from desired spacing ---
    width_m = W * pixel_w_m
    height_m = H * pixel_h_m

    x_spacing_m = x_spacing_km * 1000.0
    y_spacing_m = y_spacing_km * 1000.0

    x_dimension = max(1, int(np.ceil(width_m / x_spacing_m)))
    y_dimension = max(1, int(np.ceil(height_m / y_spacing_m)))

    print(
        "Grid configuration from spacing:\n"
        f"  Desired spacing: {x_spacing_km} km (x), {y_spacing_km} km (y)\n"
        f"  Approx grid size: {x_dimension} x {y_dimension}"
    )

    # --- Build coarse demand grid ---
    block_h = H // y_dimension
    block_w = W // x_dimension

    grid = np.zeros((y_dimension, x_dimension), dtype=np.float64)

    mask_valid = np.ones_like(data, dtype=bool)
    if nodata is not None:
        mask_valid &= data != nodata
    mask_valid &= data > 0

    for i in range(y_dimension):
        for j in range(x_dimension):
            r0 = i * block_h
            r1 = (i + 1) * block_h if i < y_dimension - 1 else H
            c0 = j * block_w
            c1 = (j + 1) * block_w if j < x_dimension - 1 else W

            block = data[r0:r1, c0:c1]
            block_mask = mask_valid[r0:r1, c0:c1]

            if block_mask.any():
                grid[i, j] = block[block_mask].sum()
            else:
                grid[i, j] = 0.0

    print("Built demand grid with shape:", grid.shape)

    # --- Plot demand grid (log1p for visibility) ---
    plt.figure(figsize=(8, 5))
    plt.imshow(np.log1p(grid), origin="upper")
    plt.title("Demand Grid (log1p of cell demand)")
    plt.colorbar(label="log(1 + demand)")
    plt.tight_layout()
    plt.savefig(out_dir / "demand_grid.png", dpi=200)
    plt.close()

    # --- Grid cell center coordinates (projected) ---
    ys, xs = np.indices(grid.shape)  # (row, col)
    row_size = H / y_dimension
    col_size = W / x_dimension

    rows = (ys + 0.5) * row_size
    cols = (xs + 0.5) * col_size

    X_m, Y_m = xy(transform, rows, cols)
    X_m = np.array(X_m)
    Y_m = np.array(Y_m)

    X_km = X_m / 1000.0
    Y_km = Y_m / 1000.0

    x0_km = X_km.min()
    y0_km = Y_km.min()


    X_km_shift = X_km - x0_km
    Y_km_shift = Y_km - y0_km

    # --- Clustering input ---
    pts = np.column_stack([X_m.ravel(), Y_m.ravel()])
    weights = grid.ravel()

    nonzero = weights > 0
    pts_nz = pts[nonzero]
    w_nz = weights[nonzero]

    print(f"Non-empty grid cells: {pts_nz.shape[0]}")

    # --- Weighted KMeans ---
    print(f"Running KMeans with K={clustering_number} ...")
    kmeans = KMeans(
        n_clusters=clustering_number,
        random_state=0,
        n_init="auto",
    )
    kmeans.fit(pts_nz, sample_weight=w_nz)

    centers_m = kmeans.cluster_centers_
    labels_nz = kmeans.labels_

    centers_km = centers_m / 1000.0
    centers_km_shift = centers_km - np.array([[x0_km, y0_km]])

    # --- Total demand per cluster ---
    cluster_demands = np.zeros(clustering_number, dtype=np.float64)
    for label, w in zip(labels_nz, w_nz):
        cluster_demands[label] += w

    nonzero_cluster = cluster_demands[cluster_demands > 0]
    if nonzero_cluster.size > 0:
        min_demand = nonzero_cluster.min()
        exponent = int(np.floor(np.log10(min_demand)))
        scale_factor = 10.0 ** exponent
    else:
        min_demand = 1.0
        exponent = 0
        scale_factor = 1.0

    print(
        "Cluster demand scaling: divide by "
        f"10^{exponent} (min non-zero demand ≈ {min_demand:.3g})"
    )

    cluster_demands_scaled = cluster_demands / scale_factor

    # --- Full label array for grid cells ---
    cluster_ids_full = np.full(weights.shape, -1, dtype=int)
    cluster_ids_full[nonzero] = labels_nz
    cluster_ids_full = cluster_ids_full.reshape(grid.shape)

    # --- Save cluster summary CSV ---
    clusters_csv = out_dir / "clusters.csv"
    cluster_ids = np.arange(clustering_number, dtype=int)

    cluster_table = np.column_stack([
        cluster_ids,
        centers_m[:, 0],
        centers_m[:, 1],
        centers_km[:, 0],
        centers_km[:, 1],
        centers_km_shift[:, 0],
        centers_km_shift[:, 1],
        cluster_demands,
        cluster_demands_scaled,
    ])

    header = (
        "cluster_id,x_m,y_m,x_km,y_km,"
        "x_km_shift,y_km_shift,"
        "total_demand,scaled_demand"
    )
    np.savetxt(clusters_csv, cluster_table, delimiter=",", header=header, comments="")
    print("Saved cluster summary to:", clusters_csv)

    combined_map = "demand_estimates.csv"
    np.savetxt(combined_map, cluster_table, delimiter=",", header=header, comments="")

    # --- Save per-grid-cell CSV ---
    grid_csv = out_dir / "grid_points.csv"
    arr_out = np.column_stack([
        ys.ravel(),
        xs.ravel(),
        X_m.ravel(),
        Y_m.ravel(),
        X_km.ravel(),
        Y_km.ravel(),
        X_km_shift.ravel(),
        Y_km_shift.ravel(),
        weights,
        cluster_ids_full.ravel(),
    ])
    header2 = (
        "i,j,x_m,y_m,x_km,y_km,"
        "x_km_shift,y_km_shift,"
        "demand,cluster_id"
    )
    np.savetxt(grid_csv, arr_out, delimiter=",", header=header2, comments="")
    print("Saved grid points to:", grid_csv)

    fig, ax = plt.subplots(figsize=(11, 8))

    if cluster_demands.max() > 0:
        demand_norm = cluster_demands / cluster_demands.max()
    else:
        demand_norm = cluster_demands
    sizes = 50 + 450 * demand_norm
    colors = cluster_demands_scaled

    sc = ax.scatter(
        centers_km_shift[:, 0],
        centers_km_shift[:, 1],
        s=sizes,
        c=colors,
        cmap="cividis",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (km, shifted)")
    ax.set_ylabel("Y (km, shifted)")
    ax.set_title(f"Cluster Centers (size & color = demand)\nRun ID: {run_id}")

    # Colorbar that is exactly as tall as the map:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label(f"Total cluster demand / 10^{exponent}")

    fig.tight_layout()
    fig.savefig(out_dir / "cluster_centers.png", dpi=200)
    plt.close(fig)

    print("Saved cluster centers plot to:", out_dir / "cluster_centers.png")

    with open(out_dir / "coord_shift.txt", "w") as f:
        f.write(f"Run ID: {run_id}\n\n")
        f.write(f"Input raster: {tif_path}\n")
        f.write(f"Original size: {W} x {H}\n")
        f.write(f"Grid shape: {grid.shape}\n")
        f.write(f"X spacing (km): {x_spacing_km}\n")
        f.write(f"Y spacing (km): {y_spacing_km}\n")
        f.write(f"Pixel size (m): {pixel_w_m:.3f} x {pixel_h_m:.3f}\n")
        f.write(f"Clustering number: {clustering_number}\n")
        f.write(f"Cluster demand scale factor: 10^{exponent}\n")
        f.write("\nGraph coordinate transformation:\n")
        f.write(f"x_graph_km = x_real_km - {x0_km:.6f}\n")
        f.write(f"y_graph_km = y_real_km - {y0_km:.6f}\n")

    return centers_km_shift, cluster_demands, cluster_ids_full, x0_km, y0_km, run_id

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage:")
        print(
            "  python demand_clustering.py "
            "<input_5070_tight.tif> <x_spacing_km> <y_spacing_km> <clustering_number>"
        )
        print("Example:")
        print(
            "  python demand_clustering.py "
            "nightlights_conus_5070_tight.tif 50 50 20"
        )
        sys.exit(1)

    tif = sys.argv[1]
    x_spacing_km = float(sys.argv[2])
    y_spacing_km = float(sys.argv[3])
    k = int(sys.argv[4])

    build_clusters(tif, x_spacing_km, y_spacing_km, k)