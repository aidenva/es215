#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import xy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def build_clusters(
    tif_path,
    x_dimension,
    y_dimension,
    clustering_number,
    out_prefix=None,
):
    """
    From a reprojected (EPSG:5070), tightly clipped CONUS raster, do:

    1) Aggregate into an X×Y grid to reduce resolution.
    2) Treat each non-zero grid cell as a sample point with a weight
       equal to its summed demand.
    3) Run weighted KMeans to get `clustering_number` clusters.
    4) For each cluster, compute:
         - cluster center (X_km, Y_km) in EPSG:5070 (km)
         - total demand assigned to that cluster.

    Outputs:
      - <out_prefix>_clusters.csv:
          cluster_id,x_m,y_m,x_km,y_km,total_demand,scaled_demand
      - <out_prefix>_grid_points.csv:
          i,j,x_m,y_m,x_km,y_km,demand,cluster_id
      - <out_prefix>_demand_grid.png: heatmap of demand on grid (log1p for visibility)
      - <out_prefix>_cluster_centers.png: cluster centers, colorbar = scaled demand
    """

    tif_path = Path(tif_path)
    if out_prefix is None:
        out_prefix = tif_path.with_suffix("")
    else:
        out_prefix = Path(out_prefix)

    # --- Load raster ---
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        transform = src.transform
        nodata = src.nodata
        crs = src.crs
        H, W = data.shape

    print(f"Loaded {tif_path}")
    print(f"  CRS: {crs}")
    print(f"  Size: {W} x {H}")

    # --- Build coarse demand grid ---
    block_h = H // y_dimension
    block_w = W // x_dimension

    grid = np.zeros((y_dimension, x_dimension), dtype=np.float64)

    # Valid pixels: >0 and not nodata
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

    # --- Plot demand grid (still log1p for visibility) ---
    plt.figure(figsize=(8, 5))
    plt.imshow(np.log1p(grid), origin="upper")
    plt.title("Demand Grid (log1p of cell demand)")
    plt.colorbar(label="log(1 + demand)")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_demand_grid.png", dpi=200)
    plt.close()

    # --- Build sample points (grid cell centers in projected coords) ---
    ys, xs = np.indices(grid.shape)  # grid indices (row, col)

    row_size = H / y_dimension
    col_size = W / x_dimension

    # Row/col index in original raster corresponding to each grid cell center
    rows = (ys + 0.5) * row_size
    cols = (xs + 0.5) * col_size

    # Convert to projected coordinates (meters) using the raster transform
    X_m, Y_m = xy(transform, rows, cols)
    X_m = np.array(X_m)
    Y_m = np.array(Y_m)

    # Also in kilometers for distance interpretation / plotting
    X_km = X_m / 1000.0
    Y_km = Y_m / 1000.0

    # Flatten for clustering (in meters → better to keep high precision)
    pts = np.column_stack([X_m.ravel(), Y_m.ravel()])
    weights = grid.ravel()

    # Keep only cells with demand > 0
    nonzero = weights > 0
    pts_nz = pts[nonzero]
    w_nz = weights[nonzero]

    print(f"Non-empty grid cells: {pts_nz.shape[0]}")

    # --- Weighted KMeans clustering ---
    print(f"Running KMeans with K={clustering_number} ...")
    kmeans = KMeans(
        n_clusters=clustering_number,
        random_state=0,
        n_init="auto",
    )
    kmeans.fit(pts_nz, sample_weight=w_nz)

    centers_m = kmeans.cluster_centers_  # shape (K, 2) in meters (EPSG:5070)
    labels_nz = kmeans.labels_           # cluster index per nonzero cell

    # Centers in km for convenience / plotting
    centers_km = centers_m / 1000.0

    # --- Compute total demand per cluster ---
    cluster_demands = np.zeros(clustering_number, dtype=np.float64)
    for label, w in zip(labels_nz, w_nz):
        cluster_demands[label] += w

    # --- Scaling factor based on smallest non-zero cluster demand ---
    nonzero_cluster = cluster_demands[cluster_demands > 0]
    if nonzero_cluster.size > 0:
        min_demand = nonzero_cluster.min()
        # Order of magnitude of smallest non-zero demand
        exponent = int(np.floor(np.log10(min_demand)))
        scale_factor = 10.0 ** exponent
    else:
        min_demand = 1.0
        exponent = 0
        scale_factor = 1.0

    print(f"Cluster demand scaling: divide by 10^{exponent} (min non-zero demand ≈ {min_demand:.3g})")

    cluster_demands_scaled = cluster_demands / scale_factor

    # --- Build full label array (for export / optional plotting) ---
    cluster_ids_full = np.full(weights.shape, -1, dtype=int)
    cluster_ids_full[nonzero] = labels_nz
    cluster_ids_full = cluster_ids_full.reshape(grid.shape)

    # --- Save cluster summary (main output) ---
    clusters_csv = f"{out_prefix}_clusters.csv"
    cluster_ids = np.arange(clustering_number, dtype=int)

    # columns: cluster_id, x_m, y_m, x_km, y_km, total_demand, scaled_demand
    cluster_table = np.column_stack([
        cluster_ids,
        centers_m[:, 0],
        centers_m[:, 1],
        centers_km[:, 0],
        centers_km[:, 1],
        cluster_demands,
        cluster_demands_scaled,
    ])

    header = "cluster_id,x_m,y_m,x_km,y_km,total_demand,scaled_demand"
    np.savetxt(clusters_csv, cluster_table, delimiter=",", header=header, comments="")
    print("Saved cluster summary to:", clusters_csv)

    # --- Save per-grid-cell assignment (optional but useful) ---
    grid_csv = f"{out_prefix}_grid_points.csv"
    # Flattened arrays: i,j,x_m,y_m,x_km,y_km,demand,cluster_id
    arr_out = np.column_stack([
        ys.ravel(),               # i (row)
        xs.ravel(),               # j (col)
        X_m.ravel(),              # x_m
        Y_m.ravel(),              # y_m
        X_km.ravel(),             # x_km
        Y_km.ravel(),             # y_km
        weights,                  # demand
        cluster_ids_full.ravel(), # cluster_id (-1 if zero demand)
    ])
    header2 = "i,j,x_m,y_m,x_km,y_km,demand,cluster_id"
    np.savetxt(grid_csv, arr_out, delimiter=",", header=header2, comments="")
    print("Saved grid points to:", grid_csv)

    # --- Plot cluster centers sized & colored by (scaled) demand ---
    plt.figure(figsize=(7, 5))

    # Marker sizes proportional to raw cluster demand (not scaled)
    if cluster_demands.max() > 0:
        demand_norm = cluster_demands / cluster_demands.max()
    else:
        demand_norm = cluster_demands
    sizes = 50 + 450 * demand_norm  # base size 50, up to 500

    # Colors use scaled demand (linear, not log)
    colors = cluster_demands_scaled

    plt.scatter(
        centers_km[:, 0],
        centers_km[:, 1],
        s=sizes,
        c=colors,
        cmap="viridis",
        edgecolor="k",
    )

    plt.title("Cluster Centers (size & color = demand)")
    cbar = plt.colorbar()
    cbar.set_label(f"Total cluster demand / 10^{exponent}")

    plt.xlabel("X (km, EPSG:5070)")
    plt.ylabel("Y (km, EPSG:5070)")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_cluster_centers.png", dpi=200)
    plt.close()

    print("Saved cluster centers plot to:", f"{out_prefix}_cluster_centers.png")

    return centers_km, cluster_demands, cluster_ids_full


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage:")
        print("  python clusters_from_raster.py <input_5070_tight.tif> <x_dim> <y_dim> <clustering_number>")
        print("Example:")
        print("  python clusters_from_raster.py nightlights_conus_5070_tight.tif 60 40 20")
        sys.exit(1)

    tif = sys.argv[1]
    x_dim = int(sys.argv[2])
    y_dim = int(sys.argv[3])
    k = int(sys.argv[4])

    build_clusters(tif, x_dim, y_dim, k)