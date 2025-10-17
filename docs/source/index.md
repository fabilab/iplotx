# iplotx documentation

```{grid} 4
:gutter: 1

  :::{grid-item}
    [![ex1](./_images/sphx_glr_plot_edit_artists_001.png)](gallery/style/plot_ports.rst)
  :::
  :::{grid-item}
    [![ex2](./_images/sphx_glr_plot_animation_thumb.gif)](gallery/other/plot_animation.rst)
  :::
  :::{grid-item}
    [![ex3](./_images/sphx_glr_plot_3d_thumb.png)](gallery/basic/plot_3d.rst)
  :::
  :::{grid-item}
    [![ex4](./_images/sphx_glr_plot_tree_node_background_thumb.png)](gallery/tree/plot_tree_node_background.rst)
  :::
```

[iplotx](https://github.com/fabilab/iplotx) is a Python library to display graphs/networks and trees with [matplotlib](https://matplotlib.org/). It natively supports [networkx](https://networkx.org/), [igraph](https://python.igraph.org/), and [graph-tool](https://graph-tool.skewed.de/) networks and [biopython](https://biopython.org/), [scikit-bio](https://scikit.bio/), [cogent3](https://cogent3.org/), [ETE4](https://etetoolkit.github.io/ete/), and [dendropy](https://jeetsukumaran.github.io/DendroPy/index.html) trees. It can also plot networks and trees from simple Python data structures for zero-dependency visualisation.

`iplotx` guarantees the **exact same visual appearance** independently of what library you used to construct the network/tree.

```{toctree}
:maxdepth: 1
:titlesonly:
:hidden:

installing
gallery/index
style
API <api>
Complete style specification <api/complete_style_specification>
Data providers <providers>
Plotting API <api/plotting>
Style API <api/style>
Artist hierarchy <api/artists>
Data provider protocols <api/providers>
```
