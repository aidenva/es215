This module contains the necessary files to generate demand clusters, given a TIF input file and a SHP file.

Step 1: Locate demands
Run:
python3 real_map.py real.tif tl_2025_us_state.shp us_map.tif f

real_map.py is the Python script
real.tif is a map that contains the region of interest (here is US)
tl_2025_us_state.shp is a Shape file. Note that they come with the other files of the same name but different extension
us_map.tif is the desired output name
f indicates that the Shape file does not filter out continental US. We then filter for only states/regions in the continental US.

Then to reshape this nicely into a linear distance map, run:
python3 reproject.py us_map.tif us_map_5070.tif

reproject.py
us_map.tif is the output from the previous code
us_map_5070.tif is the output name from this code

Step 2: Demand clustering
Run:
python3 demand_clustering.py us_map_5070.tif 10 10 20

demand_clustering.py is the Python script
us_map_5070.tif is the input from the previous code
10 10 represent the x-coordinate spacing and y-coordinate spacing in kilometers
20 is the number of clusters you want to make

This code will automatically:
- make a best attempt at clustering demand from the given map, given the number of nodes
- scale the values to some common 10^k value
- shift the location of the coordinates so that (0,0) is on the bottom left (qualitatively)

The output of this script includes:
- Two png files: demand_grid is a visualization of log-scale demand of the pre-clustered map, cluster_centers is a visuazation of demand scaled down by some factor 10^k post-clustering
- Two csv files: clusters has the cluster information. grid_points is the data used to make cluster, so there are a lot of points and is most likely not used
- One txt file: important text outputs for bookkeeping, especially with the translation

Because the clusters.csv is the most important part, I have made it so that everytime this is run, we get a file "demand_estimates.csv" in the main folder. This can always be manually changed.

This is the end of the demand part.

---

Supply map was done by eyeballing for now. To proceed, we need a csv file of a similar format as the clusters csv file, with one more column stating the type of source (wind/solar for now).

---

Now to showcase the combined map of demand and supply, run:
python3 combined_real_map.py demand_estimates.csv supply_estimates.csv tl_2025_us_state.shp

There are two versions of this, with or without the last argument, depending on whether you want the map to be overlaid.

---

Optimized flow module: Optional

This part I am building in case we are explicitly setting the connections and their capacity
Run:
python3 optimized_flow.py demand_estimates.csv supply_estimates.csv scale4 4
or:
python3 optimized_flow.py demand_estimates.csv supply_estimates.csv noscale

The logic of this module is:
- We have some demand map and clusters that we think are good proxies for demand
- We have eyeballed supply map and clusters that have physical basis for the value of supply, but the unit (billion kWh/year) are not consistent with whatever scaled value of demand is
- Therefore, we can either scale the supply by some number such that the total scaled supply and scaled demand match
- Or we scale the supply by a fixed amount and compute the map

The algorithm in optimized_flow.py is a linear programming module that tells us if there is a map that can satisfy static demand = supply, and if true minimize the distance (specifically, the distance-weighted flow) between supply and demand nodes. The distance is why we need this specific projection, and the fact that the cost is linear in the flow amount why we can do linear programming. This is perhaps why this module is only used to pre-determine edges in the graph, and not necessarily model real transport.

The outputs will include:
- A .png map illustrating the nodes and flows
- A .txt file detailing the important paramters used and computed
- A .csv file detailing the edges (which node is connected to which node)