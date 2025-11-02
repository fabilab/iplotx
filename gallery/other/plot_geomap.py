"""
Aftican cities
==============

This example shows how to use ``iplotx`` with ``cartopy`` for geodata.

.. tip::
  This also shows that ``iplotx`` works in non-cartesian coordinate systems.
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import networkx as nx
import pandas as pd
import iplotx as ipx


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([-20, 60, -40, 45], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)

layout = pd.read_csv(
    "data/african_cities_coords.csv",
    sep=",",
    index_col=0,
)[["Lon", "Lat"]]

g = nx.from_edgelist([
    ("Tunis", "Tripoli"),
    ("Tripoli", "Yaoundé"),
    ("Yaoundé", "Kinshasa"),
    ("Kinshasa", "Luanda"),
    ("Kinshasa", "Nairobi"),
    ("Nairobi", "Addis Ababa"),
    ("Addis Ababa", "Khartoum"),
    ("Khartoum", "Cairo"),
    ("Cairo", "Tunis"),
    ("Lusaka", "Harare"),
    ("Harare", "Maputo"),
    ("Maputo", "Pretoria"),
    ("Pretoria", "Luanda"),
])

ipx.network(
    g,
    layout.loc[list(g.nodes)],
    ax=ax,
    vertex_edgecolor="black",
    vertex_facecolor="tomato",
    vertex_alpha=0.8,
    vertex_label_color="black",
    vertex_size=10,
    title="Network of African Cities",
    edge_curved=True,
    edge_tension=1,
)
