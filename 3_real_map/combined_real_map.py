import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def map_demand_supply(demand_csv, supply_csv):
    demand_df = pd.read_csv(demand_csv)
    supply_df = pd.read_csv(supply_csv)

    # Split supply into wind vs solar
    wind = supply_df[supply_df["node_type"] == "wind"]
    solar = supply_df[supply_df["node_type"] == "solar"]

    # Shared color mapping for supply
    supply_vals = supply_df["scaled_supply"].values
    norm = plt.Normalize(vmin=supply_vals.min(), vmax=supply_vals.max())
    cmap = plt.cm.Blues

    fig = plt.figure(figsize=(11, 8))
    gs = gridspec.GridSpec(
        1, 3,
        width_ratios=[10, 0.5, 0.5],
        wspace=0.25
    )

    ax = fig.add_subplot(gs[0, 0])

    # === Demand nodes ===
    d_scatter = ax.scatter(
        demand_df["x_km_shift"], demand_df["y_km_shift"],
        s=20 + 5 * demand_df["scaled_demand"],
        c=demand_df["scaled_demand"],
        cmap="Reds",
        marker="o"
    )

    # === Wind supply nodes (black edge) ===
    ax.scatter(
        wind["x_km_shift"], wind["y_km_shift"],
        s=20 + 5 * wind["scaled_supply"],
        c=cmap(norm(wind["scaled_supply"])),
        marker="o",
        edgecolor="#ff00ff",
        linewidth=1.5
    )

    # === Solar supply nodes (green edge) ===
    ax.scatter(
        solar["x_km_shift"], solar["y_km_shift"],
        s=20 + 5 * solar["scaled_supply"],
        c=cmap(norm(solar["scaled_supply"])),
        marker="o",
        edgecolor="#00bf00",
        linewidth=1.5
    )

    # === Axis formatting ===
    ax.set_xlabel("x (km, shifted)")
    ax.set_ylabel("y (km, shifted)")
    ax.set_title("Supply (wind/solar) and Demand Nodes")
    ax.set_aspect("equal", "box")

    # === Colorbars ===
    # Demand
    cax1 = fig.add_subplot(gs[0, 1])
    cbar1 = fig.colorbar(d_scatter, cax=cax1)
    cbar1.set_label("Demand (scaled)")

    # Supply (shared color scale)
    cax2 = fig.add_subplot(gs[0, 2])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(supply_vals)
    cbar2 = fig.colorbar(sm, cax=cax2)
    cbar2.set_label("Supply (scaled)")

    plt.tight_layout()
    plt.savefig("demand_supply_map.png", dpi=300)
    print("Saved demand and supply map to: demand_supply_map.png")


if __name__ == "__main__":
    demand_csv = sys.argv[1] if len(sys.argv) > 1 else "demand_nodes.csv"
    supply_csv = sys.argv[2] if len(sys.argv) > 2 else "supply_nodes.csv"
    map_demand_supply(demand_csv, supply_csv)