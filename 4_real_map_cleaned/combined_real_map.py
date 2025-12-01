import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import geopandas as gpd
from shapely.ops import transform

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

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(
        1, 3,
        width_ratios=[11, 0.5, 0.5],
        wspace=0.25
    )

    ax = fig.add_subplot(gs[0, 0])

    # Demand nodes
    d_scatter = ax.scatter(
        demand_df["x_km_shift"], demand_df["y_km_shift"],
        s=40 + 10 * demand_df["scaled_demand"],
        c=demand_df["scaled_demand"],
        cmap="Reds",
        marker="o"
    )

    # Wind supply nodes (black edge)
    ax.scatter(
        wind["x_km_shift"], wind["y_km_shift"],
        s=40 + 10 * wind["scaled_supply"],
        c=cmap(norm(wind["scaled_supply"])),
        marker="o",
        edgecolor="#ff00ff",
        linewidth=1.5
    )

    # Solar supply nodes (green edge)
    ax.scatter(
        solar["x_km_shift"], solar["y_km_shift"],
        s=40 + 10 * solar["scaled_supply"],
        c=cmap(norm(solar["scaled_supply"])),
        marker="o",
        edgecolor="#00bf00",
        linewidth=1.5
    )

    ax.set_xlabel("x (km, shifted)")
    ax.set_ylabel("y (km, shifted)")
    ax.set_title("Supply (wind/solar) and Demand Nodes")
    ax.set_aspect("equal", "box")

    # Demand color scale
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

def meter_to_shifted_km(x, y, z=None):
    return x / 1000.0 + X_SHIFT, y / 1000.0 + Y_SHIFT

def map_demand_supply_overlay(demand_csv, supply_csv, shp_path):
    demand_df = pd.read_csv(demand_csv)
    supply_df = pd.read_csv(supply_csv)
    
    # --- Load shapefile and ensure EPSG:5070 ---
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None or gdf.crs.to_epsg() != 5070:
        gdf = gdf.to_crs(epsg=5070)

    lower48 = [
    "AL","AR","AZ","CA","CO","CT","DE","FL","GA","IA","ID","IL","IN","KS","KY",
    "LA","MA","MD","ME","MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ","NM",
    "NV","NY","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VA","VT","WA",
    "WI","WV","WY","DC"
    ]

    gdf = gdf[gdf["STUSPS"].isin(lower48)]

    # --- Transform shapefile geometry to shifted km coordinates ---
    gdf_shifted = gdf.copy()
    gdf_shifted["geometry"] = gdf_shifted.geometry.apply(
        lambda geom: transform(meter_to_shifted_km, geom)
    )

    # Split supply into wind vs solar
    wind = supply_df[supply_df["node_type"] == "wind"]
    solar = supply_df[supply_df["node_type"] == "solar"]

    # Shared color mapping for supply
    supply_vals = supply_df["scaled_supply"].values
    norm = plt.Normalize(vmin=supply_vals.min(), vmax=supply_vals.max())
    cmap = plt.cm.Blues

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = gridspec.GridSpec(
        1, 3,
        width_ratios=[11, 0.5, 0.5],
        wspace=0.25
    )

    ax = fig.add_subplot(gs[0, 0])
    ax.grid(False)

    gdf_shifted.plot(
        ax=ax,
        edgecolor="grey",
        facecolor="none",
        linewidth=0.5,
        alpha=0.5,     # <- faint
        zorder=0
    )

    d_scatter = ax.scatter(
        demand_df["x_km_shift"], demand_df["y_km_shift"],
        s=40 + 10 * demand_df["scaled_demand"],
        c=demand_df["scaled_demand"],
        cmap="Reds",
        marker="o",
        zorder = 2
    )

    # Wind supply nodes (black edge)
    ax.scatter(
        wind["x_km_shift"], wind["y_km_shift"],
        s=40 + 10 * wind["scaled_supply"],
        c=cmap(norm(wind["scaled_supply"])),
        marker="o",
        edgecolor="#ff00ff",
        linewidth=1.5,
        zorder = 3
    )

    # Solar supply nodes (green edge)
    ax.scatter(
        solar["x_km_shift"], solar["y_km_shift"],
        s=40 + 10 * solar["scaled_supply"],
        c=cmap(norm(solar["scaled_supply"])),
        marker="o",
        edgecolor="#00bf00",
        linewidth=1.5,
        zorder = 3
    )

    ax.set_xlabel("x (km, shifted)")
    ax.set_ylabel("y (km, shifted)")
    ax.set_title("Supply (wind/solar) and Demand Nodes")
    ax.set_aspect("equal", "box")

    # Demand color scale
    cax1 = fig.add_subplot(gs[0, 1])
    cbar1 = fig.colorbar(d_scatter, cax=cax1)
    cbar1.set_label("Demand (scaled)")

    # Supply (shared color scale)
    cax2 = fig.add_subplot(gs[0, 2])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(supply_vals)
    cbar2 = fig.colorbar(sm, cax=cax2)
    cbar2.set_label("Supply (scaled)")

    plt.savefig("demand_supply_map_us.png", dpi=300)
    print("Saved demand and supply map to: demand_supply_map_us.png")
    print("Shifted shapefile bounds:", gdf_shifted.total_bounds)

if __name__ == "__main__":
    X_SHIFT = 2917.211383  # km
    Y_SHIFT = -156.911498  # km

    if len(sys.argv) == 3:
        demand_csv = sys.argv[1]
        supply_csv = sys.argv[2]
        map_demand_supply(demand_csv, supply_csv)

    elif len(sys.argv) == 4:
        demand_csv = sys.argv[1]
        supply_csv = sys.argv[2]
        shp_path = sys.argv[3]
        map_demand_supply_overlay(demand_csv, supply_csv, shp_path)

    elif len(sys.argv) == 6:
        demand_csv = sys.argv[1]
        supply_csv = sys.argv[2]
        shp_path = sys.argv[3]
        x_shift = float(sys.argv[4])
        y_shift = float(sys.argv[5])
        X_SHIFT = x_shift
        Y_SHIFT = y_shift
        map_demand_supply_overlay(demand_csv, supply_csv, shp_path)

    else:
        print("Usage for basic map: python combined_real_map.py <demand_csv> <supply_csv>")
        print("Usage for overlay map: python combined_real_map.py <demand_csv> <supply_csv> <shapefile_path>")