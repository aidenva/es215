plant_scale=30
import pypsa
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle


def add_distribution_node(
        j,
        network,
        electrolysis=True,
        p_nom_combustion=1000,
        p_nom_electrolysis=1000,
        electrolysis_efficiency=0.4,
        combustion_efficiency=0.4,
        loc=(0, 0),
        generator='solar',
        gen_pmax=1000,
        gen_marg_cost=10,
        # new: periods (in hours) for time-varying profiles
        solar_period_hours=24.0,
        wind_period_hours=24.0,
        # new: amplitude scaling for the per-unit sinusoid (0–1)
        profile_amplitude_pu=1.0,
        # new: enable/disable infinite fuel store
        add_fuel_store=True,
        fuel_store_e_nom=np.inf,
        supplyscale = 10
):
    """
    Add a local distribution node to a PyPSA network with electric and fuel buses,
    and optionally include electrolysis for fuel production.

    If the network has multiple snapshots (time-varying):
      A) Adds an "infinite" fuel store on the fuel bus (if add_fuel_store=True)
      B) For a solar generator, applies a sinusoidal p_max_pu profile with
         period `solar_period_hours`.
      C) For a wind generator, applies a sinusoidal p_max_pu profile with
         period `wind_period_hours`.

    gen_pmax is used as the generator's p_nom (MW); the time-varying power is
    then p(t) = p_nom * p_max_pu(t), where p_max_pu(t) is in [0, profile_amplitude_pu].
    """

    # -----------------------------
    # 1) Electric bus
    # -----------------------------
    elec_node_name = f"DNode{j}_Elec"
    network.add(
        "Bus",
        elec_node_name,
        carrier="Electricity",
        plant=j,
        x=loc[0] - plant_scale,
        y=loc[1],
    )

    generator_name = None
    if generator == 'solar':
        generator_name = f"generator_elec_solar{j}"
        network.add(
            "Generator",
            generator_name,
            carrier="Electricity",
            p_nom=gen_pmax*supplyscale,
            bus=elec_node_name,
            marginal_cost=gen_marg_cost,
        )
    elif generator == 'wind':
        generator_name = f"generator_elec_wind{j}"
        network.add(
            "Generator",
            generator_name,
            carrier="Electricity",
            p_nom=gen_pmax*supplyscale,
            bus=elec_node_name,
            marginal_cost=gen_marg_cost,
        )

    # -----------------------------
    # 2) Fuel bus
    # -----------------------------
    fuel_node_name = f"DNode{j}_Fuel"
    network.add(
        "Bus",
        fuel_node_name,
        carrier="CH4",
        plant=j,
        x=loc[0] + plant_scale,
        y=loc[1],
    )

    # -----------------------------
    # 3) Combustion link (CH4 -> Elec)
    # -----------------------------
    link_name = f"combustion_distrib{j}"
    network.add(
        "Link",
        link_name,
        p_nom=p_nom_combustion,
        efficiency=combustion_efficiency,
        bus0=fuel_node_name,
        bus1=elec_node_name,
        carrier="CH4",
        conversion=True,
    )

    # -----------------------------
    # 4) Electrolysis link (Elec -> CH4)
    # -----------------------------
    if electrolysis:
        electrolysis_name = f"electrolysis_Distrib{j}"
        network.add(
            "Link",
            electrolysis_name,
            p_nom=p_nom_electrolysis,
            efficiency=electrolysis_efficiency,
            bus0=elec_node_name,
            bus1=fuel_node_name,
            carrier="Electricity",
            conversion=True,
        )

    # ------------------------------------------------------------------
    # 5) Time-varying behavior: only if network has multiple snapshots
    # ------------------------------------------------------------------
    if len(network.snapshots) > 1:

        # A) "Infinite" fuel reservoir attached to the fuel bus
        if add_fuel_store:
            fuel_store_name = f"fuel_store_DNode{j}"
            network.add(
                "Store",
                fuel_store_name,
                bus=fuel_node_name,
                e_nom=fuel_store_e_nom,  # np.inf or a very large number
                e_min_pu=0.0,
                e_max_pu=1.0,
                capital_cost=0.0,
                marginal_cost=0.0,
                e_initial=np.nan
            )

        # If no generator was added, nothing to do for profiles
        if generator_name is None:
            return

        # Build a time vector in hours relative to the first snapshot
        snaps = network.snapshots


        # If snapshots are just integers or something similar
 

        # We’ll use a shifted sine: 0.5*(1 + sin(...)) in [0, 1],
        # then scale by amp so the final profile is in [0, amp].
        if generator == 'solar':
            omega = 2.0 * np.pi / float(solar_period_hours)
            profile = 0.5 * (1.0 + np.sin(omega * network.snapshots))

        elif generator == 'wind':
            omega = 2.0 * np.pi / float(wind_period_hours)
            profile = 0.5*(1.0 + np.sin(omega * network.snapshots))

        else:
            profile = None

        # Write profile into generators_t.p_max_pu if we have one
        if profile is not None:
            # Make sure the DataFrame has the correct index
            if "p_max_pu" not in network.generators_t:
                # In older PyPSA versions this should exist by default,
                # but just in case:
                network.generators_t["p_max_pu"] = pd.DataFrame(
                    1.0,
                    index=snaps,
                    columns=network.generators.index,
                )

            s = pd.Series(profile, index=snaps, name=generator_name)

            # if column doesn't exist yet, add it with default 1.0 then overwrite
            if generator_name not in network.generators_t.p_max_pu.columns:
                network.generators_t.p_max_pu[generator_name] = 1.0

            network.generators_t.p_max_pu.loc[:, generator_name] = s



def add_load_node(j,
                  network,
                  load=1,
                  p_nom_combustion=1000,
                  combustion_efficiency=0.2,
                  loc=(0, 0),
                  ):
    """
    Adds a load node and connects it to the *nearest* distribution node (DNode{j}_Elec / DNode{j}_Fuel)
    based on the bus coordinates in `network.buses` and the provided loc = (x, y).
    """

    # ---------------------------
    # 1. Find nearest distribution node (DNode*_Elec) to this loc
    # ---------------------------
    nearest_dnode_j = None
    min_dist2 = float("inf")  # use squared distance; no need for sqrt

    for bus_name, bus in network.buses.iterrows():
        # We only look at the electric distribution nodes
        if not (bus_name.startswith("DNode") and bus_name.endswith("_Elec")):
            continue

        bx = bus["x"]
        by = bus["y"]

        dx = bx - loc[0]
        dy = by - loc[1]
        dist2 = dx*dx + dy*dy

        if dist2 < min_dist2:
            min_dist2 = dist2

            # bus_name is like "DNode7_Elec" → extract the "7"
            core = bus_name[len("DNode"):]    # "7_Elec"
            core = core.split("_")[0]         # "7"
            nearest_dnode_j = int(core)

    if nearest_dnode_j is None:
        raise ValueError("No distribution nodes (DNode*_Elec) found in network.buses.")

    distrib_node = nearest_dnode_j  # for clarity below

    # ---------------------------
    # 2. Create the load buses
    # ---------------------------

    # Electric Bus
    elec_node_name = "LNode" + str(j) + "_Elec"
    network.add("Bus", elec_node_name,
                carrier="Electricity",
                x=loc[0] - plant_scale,
                y=loc[1])

    # Fuel Bus
    fuel_node_name = "LNode" + str(j) + "_Fuel"
    network.add("Bus", fuel_node_name,
                carrier="CH4",
                x=loc[0] + plant_scale,
                y=loc[1])


    # ---------------------------
    # 3. Electric Load
    # ---------------------------
    load_name = "load" + str(j)
    network.add("Load",
                load_name,
                p_set=load,
                carrier="Electricity",
                bus=elec_node_name)

    # ---------------------------
    # 4. Connections to the nearest distrib node
    # ---------------------------
    # Names of the distribution buses this load connects to:
    dnode_elec_bus = f"DNode{distrib_node}_Elec"
    dnode_fuel_bus = f"DNode{distrib_node}_Fuel"

    elec_link_name = "elec_load_link" + str(j)
    network.add("Link",
                elec_link_name,
                p_nom=100,
                efficiency=0.9,
                bus0=dnode_elec_bus,
                bus1=elec_node_name,
                carrier="Electricity",
                conversion=False)

    fuel_link_name = "fuel_load_link" + str(j)
    network.add("Link",
                fuel_link_name,
                p_nom=100,
                efficiency=0.9,
                bus0=dnode_fuel_bus,
                bus1=fuel_node_name,
                carrier="CH4",
                conversion=False)

    # ---------------------------
    # 5. In-home CH4 → Elec generator
    # ---------------------------
    combustionlink_name = "combustion_household" + str(j)
    network.add("Link",
                combustionlink_name,
                p_nom=p_nom_combustion,
                efficiency=combustion_efficiency,
                bus0=fuel_node_name,
                bus1=elec_node_name,
                carrier="CH4",
                conversion=True)

    
def plot_network_schematic(n):
        fig, ax = plt.subplots(dpi=400)
        carrier_colors = {
            'CH4': '#50c5db',
            'Electricity': '#e60027',
        }
        special_colors = {
            "combustion": "#000000",
            "electrolysis": "#000000"
        }

        def get_link_color(name, carrier):
            for prefix, color in special_colors.items():
                if name.startswith(prefix):
                    return color
            return carrier_colors.get(carrier, "gray")

        link_colors = [
            get_link_color(name, carrier)
            for name, carrier in zip(n.links.index, n.links.carrier)
        ]



        bus_coords = n.buses[["x", "y"]]
        mask = (
                bus_coords.index.to_series().astype(str).str.contains("DNode")
                & (n.buses["carrier"] == "CH4")
            )
        dnode_coords = bus_coords[mask]

        # Draw circles
        for bus, row in dnode_coords.iterrows():
            circ = Circle(
                (row["x"]-plant_scale, row["y"]),
                80,
                facecolor="black",    # hollow
                edgecolor="none",   # black outline
                alpha=0.25,
                zorder=0           # ensure on top of lines/links
            )
            ax.add_patch(circ)


        n.plot(ax=ax, geomap=False, link_colors=link_colors)



def build_network(distribution_nodes,
                  load_nodes,
                  electric_line_loss_decay=0.0005,
                  fuel_line_loss_decay=0.0006,
                  plotbool=False,
                  electrolysis_efficiency=0.6,
                  combustion_efficiency=0.4,
                  gen_marg_cost=10,
                  link_radius2=500,
                  time_varying=False,
                  num_days=1,
                  solar_period_hours=24.0,
                  wind_period_hours=720,
                  supplyscale = 10):
    n = pypsa.Network()
    n.add("Carrier", "CH4")
    n.add("Carrier", "Electricity")

    if time_varying:
        # 6 snapshots/day → every 4 hours → total hours = num_days * 24
        total_hours = num_days * 24

        # snapshots every 4 hours: 0, 4, 8, ...
        snapshots = list(range(0, total_hours, 4))
    else:
        # single snapshot at hour 0
        snapshots = [0]

    n.set_snapshots(snapshots)


    # -------------------------------
    # Distribution nodes + links
    # -------------------------------
    for i, distrib_node in distribution_nodes.iterrows():
        add_distribution_node(
            i+1,
            n,
            electrolysis=True,  # or distrib_node-specific if you want
            electrolysis_efficiency=electrolysis_efficiency,
            combustion_efficiency=combustion_efficiency,
            loc=distrib_node["loc"],
            generator=distrib_node['node_type'],       # 'solar' or 'wind'
            gen_pmax=distrib_node["scaled_supply"],
            gen_marg_cost=gen_marg_cost,
            # pass time-varying profile params
            solar_period_hours=solar_period_hours,
            wind_period_hours=wind_period_hours,
            # only add infinite fuel store in time-varying runs
            add_fuel_store=time_varying,
            supplyscale=supplyscale
        )

        x_i, y_i = distrib_node["loc"]

        # only consider previous nodes j < i
        for j in range(0, i):
            other = distribution_nodes.iloc[j]
            x_j, y_j = other["loc"]

            # squared distance
            dx = x_i - x_j
            dy = y_i - y_j
            dist2 = dx*dx + dy*dy

            # skip if outside radius
            if dist2 > link_radius2:
                continue

            # j+1 and i+1 are your DNode indices as before
            n.add(
                "Link",
                f"{j+1}{i+1}fuel",
                p_nom=10,
                efficiency=1,
                bus0=f"DNode{j+1}_Fuel",
                bus1=f"DNode{i+1}_Fuel",
                carrier="CH4",
                conversion=False,
            )
            n.add(
                "Link",
                f"{i+1}{j+1}fuel",
                p_nom=10,
                efficiency=1,
                bus0=f"DNode{i+1}_Fuel",
                bus1=f"DNode{j+1}_Fuel",
                carrier="CH4",
                conversion=False,
            )

            n.add(
                "Link",
                f"{j+1}{i+1}elec",
                p_nom=10,
                efficiency=1,
                bus0=f"DNode{j+1}_Elec",
                bus1=f"DNode{i+1}_Elec",
                carrier="Electricity",
                conversion=False,
            )
            n.add(
                "Link",
                f"{i+1}{j+1}elec",
                p_nom=10,
                efficiency=1,
                bus0=f"DNode{i+1}_Elec",
                bus1=f"DNode{j+1}_Elec",
                carrier="Electricity",
                conversion=False,
            )
    if time_varying:
        n.stores['e_cyclic'] = True
    # -------------------------------
    # Load nodes
    # -------------------------------
    for i, load_node in load_nodes.iterrows():
        add_load_node(
            i+1,
            n,
            load=load_node["load"],
            p_nom_combustion=load_node["p_nom_combustion"],
            combustion_efficiency=load_node["combustion_efficiency"],
            loc=load_node["loc"],
        )

    if plotbool:
        carrier_colors = {
            'CH4': '#50c5db',
            'Electricity': '#e60027',
        }
        special_colors = {
            "combustion": "#000000",
            "electrolysis": "#000000"
        }

        def get_link_color(name, carrier):
            for prefix, color in special_colors.items():
                if name.startswith(prefix):
                    return color
            return carrier_colors.get(carrier, "gray")

        link_colors = [
            get_link_color(name, carrier)
            for name, carrier in zip(n.links.index, n.links.carrier)
        ]

        n.plot(geomap=False, link_colors=link_colors)


    # -------------------------------
    # Link lengths and losses
    # -------------------------------
    bus_coords = n.buses[["x", "y"]]

    def calc_length(row):
        x0, y0 = bus_coords.loc[row.bus0]
        x1, y1 = bus_coords.loc[row.bus1]
        return np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

    n.links["length"] = n.links.apply(calc_length, axis=1)

    alpha_map = {
        "Electricity": electric_line_loss_decay,
        "CH4": fuel_line_loss_decay
    }

    mask = n.links["conversion"] == False

    for carrier, alpha in alpha_map.items():
        carrier_mask = (n.links.carrier == carrier) & mask
        n.links.loc[carrier_mask, "efficiency"] = (1-alpha)**(n.links.loc[carrier_mask, "length"])

    n.links["efficiency"] = np.clip(n.links["efficiency"], 0, 1)

    # -------------------------------
    # Solve
    # -------------------------------
    n.optimize(solver_name="highs", solver_options={"solver": "ipm"})

    # -------------------------------
    # Plot dispatch and flows for one snapshot
    # -------------------------------
    if plotbool:
        plt.figure(figsize=(10, 10))
        carrier_colors = {
            'CH4': '#50c5db',
            'Electricity': '#e60027',
        }
        special_colors = {
            "combustion": "#fe8f06",
            "electrolysis": "#c0e087"
        }

        def get_link_color(name, carrier):
            for prefix, color in special_colors.items():
                if name.startswith(prefix):
                    return color
            return carrier_colors.get(carrier, "gray")

        link_colors = [
            get_link_color(name, carrier)
            for name, carrier in zip(n.links.index, n.links.carrier)
        ]

        snapshot = n.snapshots[0]  # first time step
        flows = n.links_t.p0.loc[snapshot]  # MW at each link

        line_widths = abs(flows) / flows.abs().max() * 5

        # bus sizes from generator dispatch
        gen_p = n.generators_t.p.loc[snapshot]
        bus_sizes = gen_p.groupby(n.generators.bus).sum().reindex(
            n.buses.index,
            fill_value=0
        )
        if bus_sizes.max() > 0:
            bus_sizes = bus_sizes / bus_sizes.max() / 10.0

        n.plot(
            bus_sizes=bus_sizes,
            link_widths=line_widths,
            link_colors=link_colors,
            title=f"Dispatch and Link Flows at {snapshot}",
            geomap=False
        )

    return n

def plot_network(n, t):
        fig, ax = plt.subplots(figsize=(10, 10))
        carrier_colors = {
            'CH4': '#50c5db',
            'Electricity': '#e60027',
        }
        special_colors = {
            "combustion": "#fe8f06",
            "electrolysis": "#c0e087"
        }

        def get_link_color(name, carrier):
            for prefix, color in special_colors.items():
                if name.startswith(prefix):
                    return color
            return carrier_colors.get(carrier, "gray")

        link_colors = [
            get_link_color(name, carrier)
            for name, carrier in zip(n.links.index, n.links.carrier)
        ]

        snapshot = n.snapshots[t]  # first time step
        flows = n.links_t.p0.loc[snapshot]  # MW at each link

        line_widths = abs(flows)/1.5

        # bus sizes from generator dispatch
        gen_p = n.generators_t.p.loc[snapshot]
        bus_sizes = gen_p.groupby(n.generators.bus).sum().reindex(
            n.buses.index,
            fill_value=0
        )
        if bus_sizes.max() > 0:
            bus_sizes = bus_sizes / bus_sizes.max() / 10.0

        n.plot(
            ax=ax,
            bus_sizes=bus_sizes,
            link_widths=line_widths,
            link_colors=link_colors,
            geomap=False
        )
        return ax

def min_fuel_transmission_distance(n, snapshots=None, use_p0=True, tol=1e-3):
    """
    Metric #2: shortest distance at which fuel transmission (CH4) is actually used.

    Parameters
    ----------
    tol : float
        Minimum absolute flow [MW] to consider a link as "used" in any snapshot.

    Returns
    -------
    min_dist_used : float or np.nan
        Minimum length among CH4 transmission links that have |flow| > tol
        in at least one snapshot. np.nan if none used.
    """
    if snapshots is None:
        snapshots = n.snapshots

    flows = n.links_t.p0.loc[snapshots] if use_p0 else n.links_t.p1.loc[snapshots]

    if "length" not in n.links:
        raise ValueError("n.links['length'] not found; compute it before calling this function.")

    lengths = n.links["length"]

    # CH4 transmission links only
    fuel_trans_mask = (n.links.carrier == "CH4") & (n.links.conversion == False)
    fuel_cols = flows.columns[fuel_trans_mask]

    if len(fuel_cols) == 0:
        return np.nan

    # any snapshot with |flow| > tol → link counted as "used"
    used_mask = (flows[fuel_cols].abs() > tol).any(axis=0)
    used_links = fuel_cols[used_mask]

    if len(used_links) == 0:
        return np.nan

    min_dist_used = lengths[used_links].min()
    return float(min_dist_used)

def fuel_km_metrics(n, snapshots=None, use_p0=True):
    """
    Metric #1: fuel-km percentage.

    Returns
    -------
    fuel_share_snapshot : pd.Series
        Per-snapshot fuel share (fuel power-km / total power-km).
    overall_fuel_share : float
        Fuel share aggregated over all snapshots.
    """
    if snapshots is None:
        snapshots = n.snapshots

    # choose p0 or p1 as flow reference
    flows = n.links_t.p0.loc[snapshots] if use_p0 else n.links_t.p1.loc[snapshots]

    # make sure lengths line up with link columns
    if "length" not in n.links:
        raise ValueError("n.links['length'] not found; compute it before calling fuel_km_metrics.")

    lengths = n.links["length"].reindex(flows.columns)

    # masks: transmission-only, split by carrier
    trans_mask = (n.links.conversion == False)
    fuel_mask  = (n.links.carrier == "CH4") & trans_mask
    elec_mask  = (n.links.carrier == "Electricity") & trans_mask

    fuel_cols = flows.columns[fuel_mask]
    elec_cols = flows.columns[elec_mask]

    # power-km per snapshot
    fuel_power_km = (flows[fuel_cols].abs() * lengths[fuel_cols]).sum(axis=1)
    elec_power_km = (flows[elec_cols].abs() * lengths[elec_cols]).sum(axis=1)

    total_power_km = fuel_power_km + elec_power_km

    # per-snapshot share (NaN where no flow)
    fuel_share_snapshot = fuel_power_km / total_power_km.replace(0, np.nan)

    # aggregate over all snapshots (optionally weight by snapshot_weightings)


    total_fuel_power_km = (fuel_power_km).sum()
    total_elec_power_km = (elec_power_km).sum()

    denom = total_fuel_power_km + total_elec_power_km
    overall_fuel_share = total_fuel_power_km / denom if denom > 0 else np.nan

    return fuel_share_snapshot, overall_fuel_share

def plot_system_timeseries_dualaxis(
        n,
        ax=None,
        plot_stored_energy=True,
        plot_solar=True,
        plot_wind=True,
        plot_demand=True,
        **plot_kwargs
    ):
    """
    Plot system time series on one graph with dual axes:
      LEFT  Y-axis: Total stored energy (sum over all stores, MWh)
      RIGHT Y-axis: Solar supply, Wind supply, Total demand (MW)

    Returns
    -------
    series_dict : dict of pd.Series
    ax_left : matplotlib axis (stored energy)
    ax_right : matplotlib axis (supply, demand)
    """

    snapshots = n.snapshots

    # ------------------------------------
    # Create base axis if not provided
    # ------------------------------------
    if ax is None:
        fig, ax_left = plt.subplots()
    else:
        ax_left = ax

    # Create right-hand axis
    ax_right = ax_left.twinx()

    # Default line style config
    base_kwargs = dict(linewidth=1.8)
    base_kwargs.update(plot_kwargs)

    # ------------------------------------
    # 1) Total stored energy (MWh) on LEFT AXIS
    # ------------------------------------
    total_stored = None
    if plot_stored_energy and hasattr(n, "stores_t") and "e" in n.stores_t:
        total_stored = n.stores_t.e.sum(axis=1)
        ax_left.plot(
            snapshots/24,
            total_stored,
            color="#e60027",
            label="Stored energy (MWh)",
            **base_kwargs,
        )

    # ------------------------------------
    # 2) Generator outputs (MW) on RIGHT AXIS
    # ------------------------------------
    total_solar = None
    total_wind = None
    total_demand = None
    plt.fill_betweenx([0, -10], [0, 0], [15, 15], color="#c0e087", alpha = 0.25, zorder=-100)
    plt.fill_betweenx([0, -10], [15, 15],[30, 30], color="#fe8f06", alpha = 0.25, zorder=-100)
    # Generator supply
    if hasattr(n, "generators_t") and "p" in n.generators_t:
        gen_p = n.generators_t.p  # MW
        gens = n.generators

        # carrier-based or name-based detection
        solar_ids = gens.index[gens.carrier.str.contains("solar", case=False, na=False)]
        wind_ids  = gens.index[gens.carrier.str.contains("wind",  case=False, na=False)]

        if len(solar_ids) == 0:
            solar_ids = gens.index[gens.index.str.contains("solar", case=False)]
        if len(wind_ids) == 0:
            wind_ids = gens.index[gens.index.str.contains("wind", case=False)]

        if plot_solar and len(solar_ids) > 0:
            total_solar = gen_p[solar_ids].sum(axis=1)
            ax_right.plot(
                snapshots/24,
                total_solar,
                color="#50c5db",
                label="Solar supply (MW)",
                **base_kwargs
            )

        if plot_wind and len(wind_ids) > 0:
            total_wind = gen_p[wind_ids].sum(axis=1)
            ax_right.plot(
                snapshots/24,
                total_wind,
                color="black",
                label="Wind supply (MW)",
                **base_kwargs
            )

        

    # Demand (MW)
    if plot_demand and hasattr(n, "loads_t") and "p" in n.loads_t:
        total_demand = n.loads_t.p.sum(axis=1)
        ax_right.plot(
            snapshots/24,
            total_demand,
            color="black",
            label="Total demand (MW)",
            **base_kwargs,
            linestyle=":"
        )

    # ------------------------------------
    # Labels, titles, legends
    # ------------------------------------
    ax_left.set_xlabel("Day")
    ax_left.set_ylabel("Stored energy (MWh)", color="#e60027")
    ax_right.set_ylabel("Power (MW)", color="black")



    # -------- NEW LEGEND CODE ----------

    # Get figure object (works whether you passed ax in or not)
    fig = ax_left.get_figure()

    # Collect handles/labels from both axes
    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()

    handles = handles_left + handles_right
    labels = labels_left + labels_right

    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),  # outside, to the right
        )

        # Optional: shrink plot to make room for legend
        #fig.tight_layout(rect=[0, 0, 0.8, 1])

    return ax_left, ax_right



def search_min_supplyscale_bisection(
        build_network_fn,
        distribution_nodes,
        load_nodes,
        alpha_low,
        alpha_high,
        tol=1e-2,
        max_iter=20,
        verbose=False,
        build_kwargs=None,
    ):
    """
    Quiet, clean bisection search for minimum feasible supplyscale alpha.
    Prints nothing unless verbose=True.

    Assumptions:
      • alpha_high is feasible.
      • feasibility is monotone increasing in alpha.

    Returns:
      alpha_star, n_star
    """

    if build_kwargs is None:
        build_kwargs = {}

    def try_build(alpha):
        """Return (feasible, network)."""
        try:
            n = build_network_fn(
                distribution_nodes,
                load_nodes,
                supplyscale=alpha,
                **build_kwargs
            )
        except Exception:
            return False, None

        obj = getattr(n, "objective", None)
        feasible = (obj is not None) and np.isfinite(obj)
        return feasible, n

    # Ensure high bound works
    feas_high, n_high = try_build(alpha_high)
    if not feas_high:
        raise RuntimeError(
            f"alpha_high={alpha_high} is NOT feasible — increase alpha_high."
        )

    lo, hi = alpha_low, alpha_high
    alpha_star = hi
    n_star = n_high

    # Bisection loop
    for _ in range(max_iter):
        if hi - lo < tol:
            break

        mid = 0.5 * (lo + hi)
        feasible, n_mid = try_build(mid)

        if feasible:
            hi = mid
            alpha_star = mid
            n_star = n_mid
        else:
            lo = mid

    if verbose:
        print(f"Minimum feasible supplyscale α* ≈ {alpha_star:.4f}")

    return alpha_star, n_star

import numpy as np

import numpy as np

import numpy as np

import numpy as np

def max_electric_transmission_distance_dnodes(
    n,
    snapshots=None,
    use_p0=True,
    tol=1e-3,
    electric_carriers=["Electricity"],
):
    """
    Metric: longest distance at which electric transmission between DNode* buses is used,
    but only for networks that have some fuel (CH4) transmission at all.
    If there is no CH4 transmission in the network, return np.nan.

    Parameters
    ----------
    n : pypsa.Network
    snapshots : list-like or None
        Snapshots to consider; defaults to n.snapshots.
    use_p0 : bool
        Use n.links_t.p0 if True, n.links_t.p1 if False.
    tol : float
        Minimum absolute flow [MW] to consider a link as "used" in any snapshot.
    electric_carriers : list-like or None
        List of carrier names treated as "electric".
        If None, defaults to all carriers except CH4.

    Returns
    -------
    max_dist_used : float or np.nan
        Maximum length among electric transmission links (conversion == False)
        between DNode* buses that have |flow| > tol in at least one snapshot,
        provided the network has at least one CH4 transmission link.
        np.nan if no CH4 transmission exists or no qualifying electric links are used.
    """
    if snapshots is None:
        snapshots = n.snapshots

    # 0) Require that the network has some fuel transmission (CH4, non-conversion)
    if "carrier" not in n.links or "conversion" not in n.links:
        raise ValueError("n.links must have 'carrier' and 'conversion' columns.")
    # Identify CH4 transmission lines (exclude conversion links)
    fuel_trans_mask = (n.links.carrier == "CH4") & (n.links.conversion == False)
    fuel_links = n.links.index[fuel_trans_mask]

    # Extract their flows
    flows = n.links_t.p0.loc[snapshots] if use_p0 else n.links_t.p1.loc[snapshots]

    # Check whether any fuel link is actually used in any snapshot
    fuel_used_mask = (flows[fuel_links].abs() > tol).any(axis=0)

    # If NO fuel transmission link is used → return nan

    if not fuel_used_mask.any():
        print("rejected")
        return np.nan


    # 1) Prepare flows and lengths
    flows = n.links_t.p0.loc[snapshots] if use_p0 else n.links_t.p1.loc[snapshots]

    if "length" not in n.links:
        raise ValueError("n.links['length'] not found; compute it before calling this function.")

    lengths = n.links["length"]

    # 2) Determine electric carriers if not provided
    if electric_carriers is None:
        electric_carriers = [c for c in n.links.carrier.unique() if c != "CH4"]

    # 3) Only consider links that appear in flows
    links_in_flows = n.links.index.intersection(flows.columns)

    # Base mask: electric carriers, no conversion, present in flows
    base_mask = (
        n.links.index.isin(links_in_flows)
        & n.links.carrier.isin(electric_carriers)
        & (n.links.conversion == False)
    )

    # DNode–DNode connection requirement
    bus0 = n.links.bus0.astype(str)
    bus1 = n.links.bus1.astype(str)
    dnode_mask = bus0.str.startswith("DNode") & bus1.str.startswith("DNode")

    # Combined structural mask
    struct_mask = base_mask & dnode_mask

    candidate_links = n.links.index[struct_mask]

    if len(candidate_links) == 0:
        return np.nan

    # 4) Restrict flows to candidate links
    flows_sub = flows.loc[:, candidate_links]

    # Usage: any snapshot with |flow| > tol
    used_mask = (flows_sub.abs() > tol).any(axis=0)
    used_links = flows_sub.columns[used_mask]

    if len(used_links) == 0:
        return np.nan

    max_dist_used = lengths.loc[used_links].max()
    return float(max_dist_used)



