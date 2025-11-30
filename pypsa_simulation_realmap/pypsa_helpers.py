plant_scale=50
import pypsa
import math
import numpy as np
import matplotlib.pyplot as plt
def add_distribution_node(j,
                          network,
                          electrolysis=True,
                          p_nom_combustion=1000,
                          p_nom_electrolysis=1000,
                          electrolysis_efficiency = 0.4,
                          combustion_efficiency = 0.4,
                          loc = (0,0),
                          generator = 'solar',
                          gen_pmax = 1000,
                          gen_marg_cost = 10):
    """
    Add a local distribution node to a PyPSA network with electric and fuel buses,
    and optionally include electrolysis for fuel production.

    This function creates a pair of buses representing an electricity and a methane
    (CH₄) fuel node, connected via a combustion link (CH₄ → electricity).
    If `electrolysis=True`, it also adds an electrolysis link that converts
    electricity back into methane (electricity → CH₄), representing reversible
    energy conversion at the distribution node.

    Parameters
    ----------
    j : int or str
        Index or label for the distribution node (used in naming).
    network : pypsa.Network
        The PyPSA network object to which the buses and links will be added.
    electrolysis : bool, optional
        If True, add an electrolysis link (electricity → CH₄). Default is False.
    p_nom_combustion : float, optional
        Nominal power capacity [MW] of the combustion link. Default is 10.
    p_nom_electrolysis : float, optional
        Nominal power capacity [MW] of the electrolysis link (if added). Default is 10.
    electrolysis_efficiency : float, optional
        Conversion efficiency (0–1) from electricity to CH₄ for electrolysis. Default is 0.4.
    combustion_efficiency : float, optional
        Conversion efficiency (0–1) from CH₄ to electricity for combustion. Default is 0.4.
    loc1 : tuple of float, optional
        (x, y) coordinates of the node for plotting purposes. Default is (0, 0).

    Returns
    -------
    None
        The function modifies the provided PyPSA network in place by adding new buses and links.

    Notes
    -----
    - The created components include:
        * Bus: "DNode{j}_Elec" (Electricity)
        * Bus: "DNode{j}_Fuel" (CH₄)
        * Link: "combustion_distrib{j}" (CH₄ → Electricity)
        * Link: "electrolysis_Distrib{j}" (Electricity → CH₄, if enabled)
    - The `plant` attribute is set to `j` for all added components.
    - Link directions follow the convention of PyPSA’s `bus0` → `bus1`.
    """
    
    # Electric Bus
    elec_node_name = "DNode"+str(j)+"_Elec"
    network.add("Bus", elec_node_name, carrier="Electricity", plant = j,  x=loc[0] - plant_scale, y = loc[1])

    if generator =='solar':
        generator_name = "generator_elec_solar"+str(j)

        network.add("Generator", generator_name, carrier="Electricity", p_nom=gen_pmax, bus=elec_node_name, marginal_cost=gen_marg_cost)
    if generator =='wind':
        generator_name = "generator_elec_wind"+str(j)
        network.add("Generator", generator_name, carrier="Electricity", p_nom=gen_pmax, bus=elec_node_name, marginal_cost=gen_marg_cost)


    # Fuel Bus
    fuel_node_name ="DNode"+str(j)+"_Fuel"
    network.add("Bus", fuel_node_name, carrier="CH4", plant = j,  x=loc[0] + plant_scale, y = loc[1])

    # Combustion Link
    link_name = "combustion_distrib"+str(j)
    network.add("Link",
                link_name,
                p_nom=p_nom_combustion,
                efficiency=combustion_efficiency,
                bus0 = fuel_node_name,
                bus1 = elec_node_name,
                carrier="CH4",
                conversion = True,
                )

    # Electrolysis Link
    if electrolysis:
        electrolysis_name = "electrolysis_Distrib"+str(j)
        network.add("Link",
                    electrolysis_name,
                    p_nom=p_nom_electrolysis,
                    efficiency=electrolysis_efficiency,
                    bus1 = fuel_node_name,
                    bus0 = elec_node_name,
                    carrier="Electricity",
                    conversion = True,
                    )



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

    

def build_network(distribution_nodes,
                  load_nodes,
                  electric_line_loss_decay = 0.0005,
                  fuel_line_loss_decay = 0.0006,
                  plotbool=False,
                  electrolysis_efficiency=0.6,
                  combustion_efficiency=0.4,
                  gen_marg_cost=10,
                  link_radius2 = 500):
    n = pypsa.Network()
    n.add("Carrier", "CH4")
    n.add("Carrier", "Electricity")

    for i, distrib_node in distribution_nodes.iterrows():
        add_distribution_node(i+1,
                              n,
                              electrolysis_efficiency=electrolysis_efficiency,
                              combustion_efficiency=combustion_efficiency,
                              loc = distrib_node["loc"],
                              generator=distrib_node['node_type'],
                              gen_pmax = distrib_node["scaled_supply"],
                              gen_marg_cost=gen_marg_cost
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
                p_nom=10000,
                efficiency=1,
                bus0=f"DNode{j+1}_Fuel",
                bus1=f"DNode{i+1}_Fuel",
                carrier="CH4",
                conversion=False,
            )
            n.add(
                "Link",
                f"{i+1}{j+1}fuel",
                p_nom=10000,
                efficiency=1,
                bus0=f"DNode{i+1}_Fuel",
                bus1=f"DNode{j+1}_Fuel",
                carrier="CH4",
                conversion=False,
            )

            n.add(
                "Link",
                f"{j+1}{i+1}elec",
                p_nom=10000,
                efficiency=1,
                bus0=f"DNode{j+1}_Elec",
                bus1=f"DNode{i+1}_Elec",
                carrier="Electricity",
                conversion=False,
            )
            n.add(
                "Link",
                f"{i+1}{j+1}elec",
                p_nom=10000,
                efficiency=1,
                bus0=f"DNode{i+1}_Elec",
                bus1=f"DNode{j+1}_Elec",
                carrier="Electricity",
                conversion=False,
            )


    for i, load_node in load_nodes.iterrows():
        add_load_node(i+1,
                      n,
                      load = load_node["load"],
                      p_nom_combustion=load_node["p_nom_combustion"],
                      combustion_efficiency= load_node["combustion_efficiency"],
                      loc = load_node["loc"]
                      )

    if plotbool:
        n.plot(geomap=False)

    bus_coords = n.buses[["x", "y"]]
    def calc_length(row):
        x0, y0 = bus_coords.loc[row.bus0]
        x1, y1 = bus_coords.loc[row.bus1]
        return np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

    n.links["length"] = n.links.apply(calc_length, axis=1, )


    alpha_map = {
        "Electricity": electric_line_loss_decay,  
        "CH4": fuel_line_loss_decay      # 0.05% per km
    }

    mask = n.links["conversion"] == False\


    for carrier, alpha in alpha_map.items():
        carrier_mask = (n.links.carrier == carrier) & mask
        n.links.loc[carrier_mask, "efficiency"] = np.exp(-alpha * n.links.loc[carrier_mask, "length"])


    n.links["efficiency"] = np.clip(n.links["efficiency"], 0, 1)

    n.optimize(solver_name="highs", solver_options={"solver": "ipm"})

    if(plotbool):
        plt.figure(figsize=(10, 10))
        carrier_colors = {
            'CH4': 'tab:blue',
            'Electricity': 'tab:orange',
        }
        special_colors = {
            "combustion": "firebrick",
            "electrolysis": "magenta"
        }

        def get_link_color(name, carrier):
            for prefix, color in special_colors.items():
                if name.startswith(prefix):
                    return color
            return carrier_colors.get(carrier, "gray")

        # Map each link to its color
        link_colors = [
            get_link_color(name, carrier)
            for name, carrier in zip(n.links.index, n.links.carrier)
        ]

        snapshot = n.snapshots[0]  # pick a time step
        flows = n.links_t.p0.loc[snapshot]  # MW at each link

        # scale widths (adjust multiplier for visibility)
        line_widths = abs(flows) / flows.abs().max() * 5  

        bus_sizes = n.generators_t.p.loc[snapshot].groupby(n.generators.bus).sum().reindex(n.buses.index, fill_value=0)
        bus_sizes = bus_sizes / bus_sizes.max() /10

        n.plot(
            bus_sizes=bus_sizes,
            link_widths=abs(flows)/flows.abs().max()*5,
            link_colors = link_colors,
            title=f"Dispatch and Link Flows at {snapshot}",
            geomap=False
        )
    return n

