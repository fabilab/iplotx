"""
Module defining the main matplotlib Artist for network/tree edges, EdgeCollection.

Some supporting functions are also defined here.
"""

from typing import (
    Sequence,
    Optional,
    Any,
)
from math import pi
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib as mpl

from ..typing import (
    Pair,
    LeafProperty,
)
from ..utils.matplotlib import (
    _compute_mid_coord_and_rot,
    _stale_wrapper,
    _forwarder,
)
from ..style import (
    rotate_style,
)
from ..label import LabelCollection
from ..vertex import VertexCollection
from .arrow import EdgeArrowCollection
from .geometry import (
    _compute_loops_per_angle,
    _fix_parallel_edges_straight,
    _compute_loop_path,
    _compute_edge_path,
)


@_forwarder(
    (
        "set_clip_path",
        "set_clip_box",
        "set_snap",
        "set_sketch_params",
        "set_animated",
        "set_picker",
    )
)
class EdgeCollection(mpl.collections.PatchCollection):
    """Artist for a collection of edges within a network/tree.

    This artist is derived from PatchCollection with a few notable differences:
      - It udpdates ends of each edge based on the vertex borders.
      - It may contain edge labels as a child (a LabelCollection).
      - For directed graphs, it contains arrows as a child (an EdgeArrowCollection).

    This class is not designed to be instantiated directly but rather by internal
    iplotx functions such as iplotx.network. However, some of its methods can be
    called directly to edit edge style after the initial draw.
    """

    def __init__(
        self,
        patches: Sequence[mpl.patches.Patch],
        vertex_ids: Sequence[tuple],
        vertex_collection: VertexCollection,
        *args,
        transform: mpl.transforms.Transform = mpl.transforms.IdentityTransform(),
        arrow_transform: mpl.transforms.Transform = mpl.transforms.IdentityTransform(),
        directed: bool = False,
        style: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Initialise an EdgeCollection.

        Parameters:
            patches: A sequence (usually, list) of matplotlib `Patch`es describing the edges.
            vertex_ids: A sequence of pairs `(v1, v2)`, each defining the ids of vertices at the
                end of an edge.
            vertex_collection: The VertexCollection instance containing the Artist for the
                vertices. This is needed to compute vertex borders and adjust edges accordingly.
            transform: The matplotlib transform for the edges, usually transData.
            arrow_transform: The matplotlib transform for the arrow patches. This is not the
                *offset_transform* of arrows, which is set equal to the edge transform (previous
                parameter). Instead, it specifies how arrow size scales, similar to vertex size.
                This is usually the identity transform.
            directed: Whether the graph is directed (in which case arrows are drawn, possibly
                with zero size or opacity to obtain an "arrowless" effect).
            style: The edge style (subdictionary: "edge") to use at creation.
        """
        kwargs["match_original"] = True
        self._vertex_ids = vertex_ids

        self._vertex_collection = vertex_collection
        self._style = style if style is not None else {}
        self._labels = kwargs.pop("labels", None)
        self._directed = directed
        self._arrow_transform = arrow_transform
        if "cmap" in self._style:
            kwargs["cmap"] = self._style["cmap"]
            kwargs["norm"] = self._style["norm"]

        # NOTE: This should also set the transform
        super().__init__(patches, transform=transform, *args, **kwargs)

        # Apparenyly capstyle is lost upon collection creation
        if "capstyle" in self._style:
            self.set_capstyle(self._style["capstyle"])

        # This is important because it prepares the right flags for scalarmappable
        self.set_facecolor("none")

        if self.directed:
            self._arrows = EdgeArrowCollection(
                self,
                transform=self._arrow_transform,
            )
        if self._labels is not None:
            style = self._style.get("label", {})
            self._label_collection = LabelCollection(
                self._labels,
                style=style,
                transform=transform,
            )

        if "split" in self._style:
            self._add_subedges(
                len(patches),
                self._style["split"],
            )

        zorder = self._style.get("zorder", 2)
        self.set_zorder(zorder)

    def _add_subedges(
        self,
        nedges,
        style,
    ):
        """Add subedges to shadow the current edges."""
        segments = [np.zeros((2, 2)) for i in range(nedges)]
        kwargs = {
            "linewidths": [],
            "edgecolors": [],
            "linestyles": [],
        }
        for i in range(nedges):
            vids = self._vertex_ids[i]
            stylei = rotate_style(style, index=i, key=vids, key2=vids[-1])
            for key, values in kwargs.items():
                # iplotx uses singular style properties
                key = key.rstrip("s")
                # "color" has higher priority than "edgecolor"
                if (key == "edgecolor") and ("color" in stylei):
                    val = stylei["color"]
                else:
                    val = stylei.get(key.rstrip("s"), getattr(self, f"get_{key}")()[i])
                values.append(val)

        self._subedges = mpl.collections.LineCollection(
            segments,
            transform=self.get_transform(),
            **kwargs,
        )

        # Apparently capstyle is lost upon collection creation
        if "capstyle" in style:
            self._subedges.set_capstyle(style["capstyle"])

    def get_children(self) -> tuple:
        children = []
        if hasattr(self, "_subedges"):
            children.append(self._subedges)
        if hasattr(self, "_arrows"):
            children.append(self._arrows)
        if hasattr(self, "_label_collection"):
            children.append(self._label_collection)
        return tuple(children)

    def set_figure(self, fig) -> None:
        super().set_figure(fig)
        self._update_before_draw()
        # NOTE: This sets the correct offsets in the arrows,
        # but not the correct sizes (see below)
        self._update_labels()
        for child in self.get_children():
            # NOTE: This sets the sizes with correct dpi scaling in the arrows
            child.set_figure(fig)

    @property
    def axes(self):
        return mpl.artist.Artist.axes.__get__(self)

    @axes.setter
    def axes(self, new_axes):
        mpl.artist.Artist.axes.__set__(self, new_axes)
        for child in self.get_children():
            child.axes = new_axes

    def set_transform(self, transform: mpl.transforms.Transform) -> None:
        """Set the transform for the edges and their children."""
        super().set_transform(transform)
        if hasattr(self, "_subedges"):
            self._subedges.set_transform(transform)
        if hasattr(self, "_arrows"):
            self._arrows.set_offset_transform(transform)
        if hasattr(self, "_label_collection"):
            self._label_collection.set_transform(transform)

    @property
    def directed(self) -> bool:
        """Whether the network is directed."""
        return self._directed

    @directed.setter
    def directed(self, value: bool) -> None:
        """Setter for the directed property.

        Changing this property triggers the addition/removal of arrows from the plot.
        """
        value = bool(value)
        if self._directed != value:
            # Moving to undirected, remove arrows
            if not value:
                self._arrows.remove()
                del self._arrows
            # Moving to directed, create arrows
            else:
                self._arrows = EdgeArrowCollection(
                    self,
                    transform=self._arrow_transform,
                )

            self._directed = value
            # NOTE: setting stale to True should trigger a redraw as soon as needed
            # and that will update children. We might need to verify that.
            self.stale = True

    def set_array(self, A) -> None:
        """Set the array for cmap/norm coloring."""
        # Preserve the alpha channel
        super().set_array(A)
        # Alpha needs to be kept separately
        if self.get_alpha() is None:
            self.set_alpha(self.get_edgecolor()[:, 3])
        # This is necessary to ensure edgecolors are bool-flagged correctly
        self.set_edgecolor(None)

    def update_scalarmappable(self) -> None:
        """Update colors from the scalar mappable array, if any.

        Assign edge colors from a numerical array, and match arrow colors
        if the graph is directed.
        """
        # NOTE: The superclass also sets stale = True
        super().update_scalarmappable()
        # Now self._edgecolors has the correct colorspace values
        # NOTE: The following line should include a condition on
        # whether the arrows are allowing color matching to the
        # edges. For now, we assume that if the edges are colormapped
        # we would want the arrows to be as well.
        if hasattr(self, "_arrows") and (self._A is not None):
            self._arrows.set_colors(self.get_edgecolor())

    def get_labels(self) -> Optional[LabelCollection]:
        """Get LabelCollection artist for labels if present."""
        if hasattr(self, "_label_collection"):
            return self._label_collection
        return None

    def get_mappable(self):
        """Return mappable for colorbar."""
        return self

    def _get_adjacent_vertices_info(self):
        index = self._vertex_collection.get_index()
        index = pd.Series(
            np.arange(len(index)),
            index=index,
        ).to_dict()

        voffsets = []
        vpaths = []
        vsizes = []
        for v1, v2 in self._vertex_ids:
            # NOTE: these are in the original layout coordinate system
            # not cartesianised yet.
            offset1 = self._vertex_collection.get_layout().values[index[v1]]
            offset2 = self._vertex_collection.get_layout().values[index[v2]]
            voffsets.append((offset1, offset2))

            path1 = self._vertex_collection.get_paths()[index[v1]]
            path2 = self._vertex_collection.get_paths()[index[v2]]
            vpaths.append((path1, path2))

            # NOTE: This needs to be computed here because the
            # VertexCollection._transforms are reset each draw in order to
            # accomodate for DPI changes on the canvas
            size1 = self._vertex_collection.get_sizes_dpi()[index[v1]]
            size2 = self._vertex_collection.get_sizes_dpi()[index[v2]]
            vsizes.append((size1, size2))

        return {
            "ids": self._vertex_ids,
            "offsets": voffsets,
            "paths": vpaths,
            "sizes": vsizes,
        }

    def _update_before_draw(self, transform=None):
        """Compute paths for the edges.

        Loops split the largest wedge left open by other
        edges of that vertex. The algo is:
        (i) Find what vertices each loop belongs to
        (ii) While going through the edges, record the angles
             for vertices with loops
        (iii) Plot each loop based on the recorded angles
        """
        vinfo = self._get_adjacent_vertices_info()
        vids = vinfo["ids"]
        vcenters = vinfo["offsets"]
        vpaths = vinfo["paths"]
        vsizes = vinfo["sizes"]
        loopmaxangle = pi / 180.0 * self._style.get("loopmaxangle", 60.0)

        if transform is None:
            transform = self.get_transform()
        trans = transform.transform
        trans_inv = transform.inverted().transform

        # 1. Make a list of vertices with loops, and store them for later
        # NOTE: vinfo["loops"] can be False when we want no loops (e.g. leaf edges)
        if vinfo.get("loops", True):
            loop_vertex_dict = defaultdict(lambda: dict(indices=[], edge_angles=[]))
            for i, (v1, v2) in enumerate(vids):
                # Postpone loops (step 3)
                if v1 == v2:
                    loop_vertex_dict[v1]["indices"].append(i)

        # 2. Make paths for non-loop edges
        # NOTE: keep track of parallel edges to offset them
        parallel_edges = defaultdict(list)
        paths = []
        for i, (v1, v2) in enumerate(vids):
            # Postpone loops (step 3)
            if vinfo.get("loops", True) and (v1 == v2):
                paths.append(None)
                continue

            # Coordinates of the adjacent vertices, in data coords
            vcoord_data = vcenters[i]

            # Vertex paths in figure (default) coords
            vpath_fig = vpaths[i]

            # Vertex size
            vsize_fig = vsizes[i]

            # Leaf rotation
            edge_stylei = rotate_style(self._style, index=i, key=(v1, v2))
            if edge_stylei.get("curved", False):
                tension = edge_stylei.get("tension", 5)
                ports = edge_stylei.get("ports", (None, None))
            else:
                tension = 0
                ports = None

            # Scale shrink by dpi
            dpi = self.figure.dpi if hasattr(self, "figure") else 72.0
            shrink = dpi / 72.0 * edge_stylei.pop("shrink", 0)

            # False is a synonym for "none"
            waypoints = edge_stylei.get("waypoints", "none")
            if waypoints is False or waypoints is np.False_:
                waypoints = "none"
            elif isinstance(waypoints, (list, tuple)) and len(waypoints) == 0:
                waypoints = "none"
            elif waypoints is True or waypoints is np.True_:
                raise ValueError(
                    "Could not determine automatically type of edge waypoints.",
                )
            if waypoints != "none":
                ports = edge_stylei.get("ports", (None, None))

            # Compute actual edge path
            path, angles = _compute_edge_path(
                vcoord_data,
                vpath_fig,
                vsize_fig,
                trans,
                trans_inv,
                tension=tension,
                waypoints=waypoints,
                ports=ports,
                layout_coordinate_system=self._vertex_collection.get_layout_coordinate_system(),
                shrink=shrink,
            )

            offset = edge_stylei.get("offset", 0)
            if np.isscalar(offset):
                if offset == 0:
                    offset = (0, 0)
                else:
                    vd_fig = trans(vcoord_data[1]) - trans(vcoord_data[0])
                    vd_fig /= np.linalg.norm(vd_fig)
                    vrot = vd_fig @ np.array([[0, -1], [1, 0]])
                    offset = offset * vrot
            offset = np.asarray(offset, dtype=float)
            # Scale by dpi
            offset *= dpi / 72.0
            if (offset != 0).any():
                path.vertices[:] = trans_inv(trans(path.vertices) + offset)

            # If splitting is active, split the path here, shedding off the last straight
            # segment but only if waypoints were used
            if hasattr(self, "_subedges") and waypoints != "none":
                # NOTE: we are already in the middle of a redraw, so we can happily avoid
                # causing stale of the subedges. They are already scheduled to be redrawn
                # at the end of this function.
                self._subedges._paths[i].vertices[:] = path.vertices[-2:].copy()
                # NOTE: instead of shortening the path, we just make the last bit invisible
                # that makes it easier on memory management etc.
                path.vertices[-1] = path.vertices[-2]

            # Collect angles for this vertex, to be used for loops plotting below
            if vinfo.get("loops", True):
                if v1 in loop_vertex_dict:
                    loop_vertex_dict[v1]["edge_angles"].append(angles[0])
                if v2 in loop_vertex_dict:
                    loop_vertex_dict[v2]["edge_angles"].append(angles[1])

            # Add the path for this non-loop edge
            paths.append(path)
            # FIXME: curved parallel edges depend on the direction of curvature...!
            parallel_edges[(v1, v2)].append(i)

        # Fix parallel edges
        # If none found, empty the dictionary already
        if (len(parallel_edges) == 0) or (max(parallel_edges.values(), key=len) == 1):
            parallel_edges = {}
        if not self._style.get("curved", False):
            while len(parallel_edges) > 0:
                (v1, v2), indices = parallel_edges.popitem()
                indices_inv = parallel_edges.pop((v2, v1), [])
                ntot = len(indices) + len(indices_inv)
                if ntot > 1:
                    _fix_parallel_edges_straight(
                        paths,
                        indices,
                        indices_inv,
                        trans,
                        trans_inv,
                        paralleloffset=self._style.get("paralleloffset", 3),
                    )

        # 3. Deal with loops at the end
        if vinfo.get("loops", True):
            for vid, ldict in loop_vertex_dict.items():
                vpath = vpaths[ldict["indices"][0]][0]
                vsize = vsizes[ldict["indices"][0]][0]
                vcoord_fig = trans(vcenters[ldict["indices"][0]][0])
                nloops = len(ldict["indices"])
                edge_angles = ldict["edge_angles"]

                # The space between the existing angles is where we can fit the loops
                # One loop we can fit in the largest wedge, multiple loops we need
                nloops_per_angle = _compute_loops_per_angle(nloops, edge_angles)

                idx = 0
                for theta1, theta2, nloops in nloops_per_angle:
                    # Angular size of each loop in this wedge
                    delta = (theta2 - theta1) / nloops

                    # Iterate over individual loops
                    for j in range(nloops):
                        thetaj1 = theta1 + j * delta + max(delta - loopmaxangle, 0) / 2
                        thetaj2 = thetaj1 + min(delta, loopmaxangle)

                        # Get the path for this loop
                        path = _compute_loop_path(
                            vcoord_fig,
                            vpath,
                            vsize,
                            thetaj1,
                            thetaj2,
                            trans_inv,
                            looptension=self._style.get("looptension", 2.5),
                        )
                        paths[ldict["indices"][idx]] = path
                        idx += 1

        self._paths = paths
        # FIXME:??
        # if hasattr(self, "_subedges"):
        #    self._subedges.stale = True

    def _update_labels(self):
        if self._labels is None:
            return

        style = self._style.get("label", None) if self._style is not None else {}
        transform = self.get_transform()
        trans = transform.transform

        offsets = []
        if not style.get("rotate", True):
            rotations = []
        for path in self._paths:
            offset, rotation = _compute_mid_coord_and_rot(path, trans)
            offsets.append(offset)
            if not style.get("rotate", True):
                rotations.append(rotation)

        self._label_collection.set_offsets(offsets)
        if not style.get("rotate", True):
            self._label_collection.set_rotations(rotations)

    @_stale_wrapper
    def draw(self, renderer):
        # Visibility affects the children too
        if not self.get_visible():
            return

        # This includes the subedges if present
        self._update_before_draw()

        # Now you can draw the edges
        super().draw(renderer)

        # This sets the labels offsets
        self._update_labels()

        # Now you can draw arrows and labels
        for child in self.get_children():
            child.draw(renderer)

    def get_ports(self) -> Optional[LeafProperty[Pair[Optional[str]]]]:
        """Get the ports for all edges.

        Returns:
            The ports for the edges, as a pair of strings or None for each edge. If None, it
            means all edges are free.
        """
        return self._style.get("ports", None)

    def set_ports(self, ports: Optional[LeafProperty[Pair[Optional[str]]]]) -> None:
        """Set new ports for the edges.

        Parameters:
            ports: A pair of ports strings for each edge. Each port can be None to mean free
                edge end.
        """
        if ports is None:
            if "ports" in self._style:
                del self._style["ports"]
        else:
            self._style["ports"] = ports
        self.stale = True

    def get_tension(self) -> Optional[LeafProperty[float]]:
        """Get the tension for the edges.

        Returns:
            The tension for the edges. If None, the edges are straight.
        """
        return self._style.get("tension", None)

    def set_tension(self, tension: Optional[LeafProperty[float]]) -> None:
        """Set new tension for the edges.

        Parameters:
            tension: The tension to use for curved edges. If None, the edges become straight.

        Note: This function does not set self.set_curved(True) automatically. If you are
        unsure whether that property is set already, you should call both functions.

        Example:
            # Set curved edges with different tensions
            >>> network.get_edges().set_curved(True)
            >>> network.get_edges().set_tension([1, 0.5])

            # Set straight edges
            # (the latter call is optional but helps readability)
            >>> network.get_edges().set_curved(False)
            >>> network.get_edges().set_tension(None)

        """
        if tension is None:
            if "tension" in self._style:
                del self._style["tension"]
        else:
            self._style["tension"] = tension
        self.stale = True

    get_tensions = get_tension
    set_tensions = set_tension

    def get_curved(self) -> bool:
        """Get whether the edges are curved or not.

        Returns:
            A bool that is True if the edges are curved, False if they are straight.
        """
        return self._style.get("curved", False)

    def set_curved(self, curved: bool) -> None:
        """Set whether the edges are curved or not.

        Parameters:
            curved: Whether the edges should be curved (True) or straight (False).

        Note: If you want only some edges to be curved, set curved to True and set tensions to
        0 for the straight edges.
        """
        self._style["curved"] = bool(curved)
        self.stale = True

    def get_loopmaxangle(self) -> Optional[float]:
        """Get the maximum angle for loops.

        Returns:
            The maximum angle in degrees that a loop can take. If None, the default is 60.
        """
        return self._style.get("loopmaxangle", 60)

    def set_loopmaxangle(self, loopmaxangle: float) -> None:
        """Set the maximum angle for loops.

        Parameters:
            loopmaxangle: The maximum angle in degrees that a loop can take.
        """
        self._style["loopmaxangle"] = loopmaxangle
        self.stale = True

    def get_looptension(self) -> Optional[float]:
        """Get the tension for loops.

        Returns:
            The tension for loops. If None, the default is 2.5.
        """
        return self._style.get("looptension", 2.5)

    def set_looptension(self, looptension: Optional[float]) -> None:
        """Set new tension for loops.

        Parameters:
            looptension: The tension to use for loops. If None, the default is 2.5.
        """
        if looptension is None:
            if "looptension" in self._style:
                del self._style["looptension"]
        else:
            self._style["looptension"] = looptension
        self.stale = True

    def get_offset(self) -> Optional[float]:
        """Get the offset for parallel straight edges.

        Returns:
            The offset in points for parallel straight edges. If None, the default is 3.
        """
        return self._style.get("offset", 3)

    def set_offset(self, offset: Optional[float]) -> None:
        """Set the offset for parallel straight edges.

        Parameters:
            offset: The offset in points for parallel straight edges. If None, the default is 3.
        """
        if offset is None:
            if "offset" in self._style:
                del self._style["offset"]
        else:
            self._style["offset"] = offset
        self.stale = True


def make_stub_patch(**kwargs):
    """Make a stub undirected edge patch, without actual path information."""
    kwargs["clip_on"] = kwargs.get("clip_on", True)
    if ("color" in kwargs) and ("edgecolor" not in kwargs):
        kwargs["edgecolor"] = kwargs.pop("color")

    # Edges are always hollow, because they are not closed paths
    # NOTE: This is supposed to cascade onto what boolean flags are set
    # for color mapping (Colorizer)
    kwargs["facecolor"] = "none"

    # Forget specific properties that are not supported here
    forbidden_props = [
        "arrow",
        "label",
        "curved",
        "tension",
        "waypoints",
        "ports",
        "looptension",
        "loopmaxangle",
        "offset",
        "paralleloffset",
        "cmap",
        "norm",
        "split",
        "shrink",
        "depthshade",
        # DEPRECATED
        "padding",
    ]
    for prop in forbidden_props:
        if prop in kwargs:
            kwargs.pop(prop)

    # NOTE: the path is overwritten later anyway, so no reason to spend any time here
    art = mpl.patches.PathPatch(
        mpl.path.Path([[0, 0]]),
        **kwargs,
    )
    return art
