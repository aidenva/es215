import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from scipy.optimize import linprog
from pathlib import Path

# ------------------------------------------------------------
# 1. OPTIMIZATION: MIN-DISTANCE SUPPLY → DEMAND FLOWS
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.optimize import linprog

def solve_supply_demand_network_from_dfs(
    demand_df,
    supply_df,
    supply_scale_factor=None,
):
    """
    Solve min-distance flow from supply nodes to demand nodes using SciPy linprog
    (no external solver binary required).

    demand_df: must contain
        'cluster_id', 'x_km_shift', 'y_km_shift', 'scaled_demand'
    supply_df: must contain
        'node_id', 'x_km_shift', 'y_km_shift', 'scaled_supply'

    supply_scale_factor:
        - if None, auto-chosen so total_supply == total_demand
        - else, all scaled_supply are multiplied by this factor

    Returns:
        flows_df with columns:
            ['supply_node_id', 'demand_cluster_id', 'flow', 'distance_km']
        and meta dict.
    """

    required_demand_cols = {"cluster_id", "x_km_shift", "y_km_shift", "scaled_demand"}
    required_supply_cols = {"node_id", "x_km_shift", "y_km_shift", "scaled_supply"}

    if not required_demand_cols.issubset(demand_df.columns):
        missing = required_demand_cols - set(demand_df.columns)
        raise ValueError(f"Demand dataframe is missing columns: {missing}")

    if not required_supply_cols.issubset(supply_df.columns):
        missing = required_supply_cols - set(supply_df.columns)
        raise ValueError(f"Supply dataframe is missing columns: {missing}")

    # Indices & basic data
    demand_ids = demand_df["cluster_id"].tolist()
    supply_ids = supply_df["node_id"].tolist()

    demand = demand_df["scaled_demand"].to_numpy(dtype=float)
    base_supply = supply_df["scaled_supply"].to_numpy(dtype=float)

    total_demand = demand.sum()
    total_base_supply = base_supply.sum()

    if total_base_supply <= 0:
        raise ValueError("Total base scaled_supply must be positive.")

    # Scale factor
    if supply_scale_factor is None:
        supply_scale_factor = total_demand / total_base_supply

    effective_supply = base_supply * supply_scale_factor

    # Coordinates
    demand_coords = demand_df[["x_km_shift", "y_km_shift"]].to_numpy(dtype=float)
    supply_coords = supply_df[["x_km_shift", "y_km_shift"]].to_numpy(dtype=float)

    n_s = len(supply_ids)
    n_d = len(demand_ids)

    # Cost matrix: Euclidean distance
    cost = np.linalg.norm(
        supply_coords[:, None, :] - demand_coords[None, :, :],
        axis=2,
    )  # shape (n_s, n_d)

    # Flatten variables: x[i,j] -> index k = i*n_d + j
    c = cost.flatten()  # objective coefficients

    # -------------------------------
    # Constraints
    # -------------------------------
    # Demand constraints: sum_i x_ij = demand_j
    # A_eq shape: (n_d, n_s * n_d)
    A_eq = np.zeros((n_d, n_s * n_d))
    b_eq = demand.copy()

    for j in range(n_d):
        for i in range(n_s):
            k = i * n_d + j
            A_eq[j, k] = 1.0

    # Supply constraints: sum_j x_ij <= supply_i
    # A_ub shape: (n_s, n_s * n_d)
    A_ub = np.zeros((n_s, n_s * n_d))
    b_ub = effective_supply.copy()

    for i in range(n_s):
        for j in range(n_d):
            k = i * n_d + j
            A_ub[i, k] = 1.0

    # Bounds: x_ij >= 0
    bounds = [(0, None)] * (n_s * n_d)

    # -------------------------------
    # Solve LP
    # -------------------------------
    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    x = res.x.reshape((n_s, n_d))  # shape (n_s, n_d)

    # -------------------------------
    # Build flows dataframe
    # -------------------------------
    records = []
    for i, sid in enumerate(supply_ids):
        for j, did in enumerate(demand_ids):
            flow_val = x[i, j]
            if flow_val > 1e-9:  # keep only positive flows
                records.append(
                    {
                        "supply_node_id": sid,
                        "demand_cluster_id": did,
                        "flow": flow_val,
                        "distance_km": cost[i, j],
                    }
                )

    flows_df = pd.DataFrame(records)

    meta = {
        "supply_scale_factor": supply_scale_factor,
        "total_demand": float(total_demand),
        "total_effective_supply": float(effective_supply.sum()),
        "status": "optimal (SciPy linprog)",
    }

    return flows_df, meta

# ------------------------------------------------------------
# 2. VISUALIZATION: NODES + FLOWS
# ------------------------------------------------------------
def plot_demand_supply_with_flows(
    demand_df,
    supply_df,
    flows_df=None,
    min_rel_flow=0.01,
    output_png="demand_supply_flows.png",
    scale_factor=1.0,
):
    """
    Plot demand and supply nodes, optionally with flow lines.

    demand_df / supply_df: same format as above.
    flows_df: dataframe with ['supply_node_id', 'demand_cluster_id', 'flow'].
    min_rel_flow: ignore flows smaller than this fraction of max flow.
    """

    # Split supply into wind vs solar
    if "node_type" in supply_df.columns:
        wind = supply_df[supply_df["node_type"] == "wind"]
        solar = supply_df[supply_df["node_type"] == "solar"]
    else:
        # if node_type doesn't exist, treat all as generic supply
        wind = supply_df.iloc[0:0]  # empty
        solar = supply_df.copy()

    # Shared color mapping for supply
    supply_vals = supply_df["scaled_supply"].values
    supply_vals = supply_vals * scale_factor
    supply_norm = plt.Normalize(vmin=supply_vals.min(), vmax=supply_vals.max())
    supply_cmap = plt.cm.Blues

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(
        1,
        3,
        width_ratios=[11, 0.5, 0.5],
        wspace=0.25,
    )

    ax = fig.add_subplot(gs[0, 0])

    # DEMAND NODES
    d_scatter = ax.scatter(
        demand_df["x_km_shift"],
        demand_df["y_km_shift"],
        s=40 + 10 * demand_df["scaled_demand"],
        c=demand_df["scaled_demand"],
        cmap="Reds",
        marker="o",
        label="Demand",
    )

    # SUPPLY NODES - wind (magenta edge)
    if not wind.empty:
        ax.scatter(
            wind["x_km_shift"],
            wind["y_km_shift"],
            s=40 + 10 * wind["scaled_supply"]*scale_factor,
            c=supply_cmap(supply_norm(wind["scaled_supply"]*scale_factor)),
            marker="o",
            edgecolor="#ff00ff",
            linewidth=1.5,
            label="Wind supply",
        )

    # SUPPLY NODES - solar (green edge)
    if not solar.empty:
        ax.scatter(
            solar["x_km_shift"],
            solar["y_km_shift"],
            s=40 + 10 * solar["scaled_supply"]*scale_factor,
            c=supply_cmap(supply_norm(solar["scaled_supply"]*scale_factor)),
            marker="o",
            edgecolor="#00bf00",
            linewidth=1.5,
            label="Solar supply",
        )

    # FLOW LINES
    if flows_df is not None and not flows_df.empty:
        required_flow_cols = {"supply_node_id", "demand_cluster_id", "flow"}
        if not required_flow_cols.issubset(flows_df.columns):
            missing = required_flow_cols - set(flows_df.columns)
            raise ValueError(f"flows_df is missing columns: {missing}")

        max_flow = flows_df["flow"].max()
        if max_flow > 0:
            cutoff = max_flow * min_rel_flow
            flows_df = flows_df[flows_df["flow"] >= cutoff]

        flows_merged = (
            flows_df.merge(
                supply_df[["node_id", "x_km_shift", "y_km_shift"]],
                left_on="supply_node_id",
                right_on="node_id",
                how="left",
                suffixes=("", "_supply"),
            )
            .merge(
                demand_df[["cluster_id", "x_km_shift", "y_km_shift"]],
                left_on="demand_cluster_id",
                right_on="cluster_id",
                how="left",
                suffixes=("_supply", "_demand"),
            )
        )

        segments = []
        line_widths = []
        flows_vals = flows_merged["flow"].values

        if len(flows_vals) > 0:
            max_flow_for_lw = flows_vals.max()
            min_lw, max_lw = 0.5, 5.0  # thickness range

            for _, row in flows_merged.iterrows():
                xs = row["x_km_shift_supply"]
                ys = row["y_km_shift_supply"]
                xd = row["x_km_shift_demand"]
                yd = row["y_km_shift_demand"]
                f = row["flow"]

                if pd.isna(xs) or pd.isna(ys) or pd.isna(xd) or pd.isna(yd):
                    continue

                segments.append([(xs, ys), (xd, yd)])

                rel = f / max_flow_for_lw if max_flow_for_lw > 0 else 0
                lw = min_lw + (max_lw - min_lw) * rel
                line_widths.append(lw)

            if segments:
                lc = LineCollection(
                    segments,
                    linewidths=line_widths,
                    colors="gray",
                    alpha=0.5,
                    zorder=0,  # behind points
                )
                ax.add_collection(lc)

    # Axes, legends, colorbars
    ax.set_xlabel("x (km, shifted)")
    ax.set_ylabel("y (km, shifted)")
    title = "Supply (wind/solar) and Demand Nodes"
    if flows_df is not None and not flows_df.empty:
        title += " with Flows"
    ax.set_title(title)
    ax.set_aspect("equal", "box")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)

    # Demand color scale
    cax1 = fig.add_subplot(gs[0, 1])
    cbar1 = fig.colorbar(d_scatter, cax=cax1)
    cbar1.set_label("Demand (scaled)")

    # Supply (shared color scale)
    cax2 = fig.add_subplot(gs[0, 2])
    sm = plt.cm.ScalarMappable(norm=supply_norm, cmap=supply_cmap)
    sm.set_array(supply_vals)
    cbar2 = fig.colorbar(sm, cax=cax2)
    cbar2.set_label("Supply (scaled)")

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"Saved demand/supply map to: {output_png}")


# ------------------------------------------------------------
# 3. FULL PIPELINE: CSV → FLOWS → PLOT
# ------------------------------------------------------------
def run_full_pipeline(
    demand_csv="nodes_demand.csv",
    supply_csv="nodes_supply.csv",
    supply_scale_factor=None,      # None → auto-balance total_supply = total_demand
    min_rel_flow=0.01,
    output_png="demand_supply_flows.png",
    summary_txt="run_summary.txt",
    flows_csv="optimized_flows.csv"
):
    # Load CSVs once
    demand_df = pd.read_csv(demand_csv)
    supply_df = pd.read_csv(supply_csv)

    # Solve optimization
    flows_df, meta = solve_supply_demand_network_from_dfs(
        demand_df,
        supply_df,
        supply_scale_factor=supply_scale_factor,
    )

    # --- Build summary text ---
    summary_text = (
        f"Supply scale factor used: {meta['supply_scale_factor']}\n"
        f"Total demand: {meta['total_demand']}\n"
        f"Total effective supply: {meta['total_effective_supply']}\n"
        f"Status: {meta['status']}\n"
        f"Number of flow edges: {len(flows_df)}\n"
    )

    # Print to console
    print(summary_text)

    # --- Write the same text to a .txt file ---
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"Saved summary to: {summary_txt}")

    # --- Save flows to CSV ---
    flows_df.to_csv(flows_csv, index=False)
    print(f"Saved optimized flows to: {flows_csv}")

    # --- Produce the plot ---
    plot_demand_supply_with_flows(
        demand_df,
        supply_df,
        flows_df=flows_df,
        min_rel_flow=min_rel_flow,
        output_png=output_png,
        scale_factor=meta['supply_scale_factor'],
    )

    return flows_df, meta

if __name__ == "__main__":
    # Example usage: adapt paths / options or hook into argparse
    # run_full_pipeline(
    #     demand_csv="nodes_demand.csv",
    #     supply_csv="nodes_supply.csv",
    #     supply_scale_factor=None,        # or set a specific numeric factor
    #     min_rel_flow=0.02,               # hide very small flows
    #     output_png="demand_supply_flows.png",
    # )

    if len(sys.argv) < 4:
        print(
            "Usage: python optimized_flow.py <demand.csv> <supply.csv> <output> [supply_scale_factor]\n"
            "Example: python optimized_flow.py nodes_demand.csv nodes_supply.csv trial_name [3]"
        )

    else:
        demand_csv = sys.argv[1]
        supply_csv = sys.argv[2]
        output = sys.argv[3]
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_png = output_dir / "map.png"
        output_txt = output_dir / "summary.txt"
        output_csv = output_dir / "flows.csv"
        supply_scale_factor = None
        if len(sys.argv) >= 5:
            try:
                supply_scale_factor = float(sys.argv[4])
            except ValueError:
                print("Invalid supply_scale_factor; using None (auto-balance).")

        run_full_pipeline(
            demand_csv=demand_csv,
            supply_csv=supply_csv,
            supply_scale_factor=supply_scale_factor,
            output_png=output_png,
            summary_txt=output_txt,
            flows_csv=output_csv,
        )