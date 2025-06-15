"""
Module defining the main matplotlib Artist for network/tree edges, EdgeCollection.

Some supporting functions are also defined here.
"""

from typing import (
    Sequence,
    Optional,
    Never,
    Any,
)
from math import atan2, tan, cos, pi, sin
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib as mpl

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
from .ports import _get_port_unit_vector


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
        layout: pd.DataFrame,
        *args,
        layout_coordinate_system: str = "cartesian",
        transform: mpl.transforms.Transform = mpl.transforms.IdentityTransform(),
        arrow_transform: mpl.transforms.Transform = mpl.transforms.IdentityTransform(),
        directed: bool = False,
        style: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Never:
        """Initialise an EdgeCollection.

        Parameters:
            patches: A sequence (usually, list) of matplotlib `Patch`es describing the edges.
            vertex_ids: A sequence of pairs `(v1, v2)`, each defining the ids of vertices at the
                end of an edge.
            vertex_collection: The VertexCollection instance containing the Artist for the
                vertices. This is needed to compute vertex borders and adjust edges accordingly.
            layout: The vertex layout.
            layout_coordinate_system: The coordinate system the previous parameter is in. For
                certain layouts, this might not be "cartesian" (e.g. "polar" layour for radial
                trees).
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
        # NOTE: the layout is needed for non-cartesian coordinate systems
        # for which information is lost upon cartesianisation (e.g. polar,
        # for which multiple angles are degenerate in cartesian space).
        self._layout = layout
        self._layout_coordinate_system = layout_coordinate_system
        self._style = style if style is not None else {}
        self._labels = kwargs.pop("labels", None)
        self._directed = directed
        self._arrow_transform = arrow_transform
        if "cmap" in self._style:
            kwargs["cmap"] = self._style["cmap"]
            kwargs["norm"] = self._style["norm"]

        # NOTE: This should also set the transform
        super().__init__(patches, transform=transform, *args, **kwargs)

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

    def get_children(self) -> tuple:
        children = []
        if hasattr(self, "_arrows"):
            children.append(self._arrows)
        if hasattr(self, "_label_collection"):
            children.append(self._label_collection)
        return tuple(children)

    def set_figure(self, fig) -> Never:
        super().set_figure(fig)
        self._update_paths()
        # NOTE: This sets the correct offsets in the arrows,
        # but not the correct sizes (see below)
        self._update_children()
        for child in self.get_children():
            # NOTE: This sets the sizes with correct dpi scaling in the arrows
            child.set_figure(fig)

    def _update_children(self):
        self._update_arrows()
        self._update_labels()

    @property
    def directed(self) -> bool:
        """Whether the network is directed."""
        return self._directed

    @directed.setter
    def directed(self, value) -> Never:
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

    def set_array(self, A) -> Never:
        """Set the array for cmap/norm coloring."""
        # Preserve the alpha channel
        super().set_array(A)
        # Alpha needs to be kept separately
        if self.get_alpha() is None:
            self.set_alpha(self.get_edgecolor()[:, 3])
        # This is necessary to ensure edgecolors are bool-flagged correctly
        self.set_edgecolor(None)

    def update_scalarmappable(self) -> Never:
        """Update colors from the scalar mappable array, if any.

        Assign edge colors from a numerical array, and match arrow colors
        if the graph is directed.
        """
        # NOTE: The superclass also sets stale = True
        super().update_scalarmappable()
        # Now self._edgecolors has the correct colorspace values
        if hasattr(self, "_arrows"):
            self._arrows.set_colors(self.get_edgecolors())

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
        )

        voffsets = []
        vpaths = []
        vsizes = []
        for v1, v2 in self._vertex_ids:
            # NOTE: these are in the original layout coordinate system
            # not cartesianised yet.
            offset1 = self._layout.values[index[v1]]
            offset2 = self._layout.values[index[v2]]
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

    def _update_paths(self, transform=None):
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
        loopmaxangle = pi / 180.0 * self._style.get("loopmaxangle", pi / 3)

        if transform is None:
            transform = self.get_transform()
        trans = transform.transform
        trans_inv = transform.inverted().transform

        # 1. Make a list of vertices with loops, and store them for later
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
            if v1 == v2:
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

            waypoints = edge_stylei.get("waypoints", "none")

            # Compute actual edge path
            path, angles = self._compute_edge_path(
                vcoord_data,
                vpath_fig,
                vsize_fig,
                trans,
                trans_inv,
                tension=tension,
                waypoints=waypoints,
                ports=ports,
            )

            # Collect angles for this vertex, to be used for loops plotting below
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
                    self._fix_parallel_edges_straight(
                        paths,
                        indices,
                        indices_inv,
                        trans,
                        trans_inv,
                        offset=self._style.get("offset", 3),
                    )

        # 3. Deal with loops at the end
        for vid, ldict in loop_vertex_dict.items():
            vpath = vpaths[ldict["indices"][0]][0]
            vsize = vsizes[ldict["indices"][0]][0]
            vcoord_fig = trans(vcenters[ldict["indices"][0]][0])
            nloops = len(ldict["indices"])
            edge_angles = ldict["edge_angles"]

            # The space between the existing angles is where we can fit the loops
            # One loop we can fit in the largest wedge, multiple loops we need
            nloops_per_angle = self._compute_loops_per_angle(nloops, edge_angles)

            idx = 0
            for theta1, theta2, nloops in nloops_per_angle:
                # Angular size of each loop in this wedge
                delta = (theta2 - theta1) / nloops

                # Iterate over individual loops
                for j in range(nloops):
                    thetaj1 = theta1 + j * delta + max(delta - loopmaxangle, 0) / 2
                    thetaj2 = thetaj1 + min(delta, loopmaxangle)

                    # Get the path for this loop
                    path = self._compute_loop_path(
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

    def _fix_parallel_edges_straight(
        self,
        paths,
        indices,
        indices_inv,
        trans,
        trans_inv,
        offset=3,
    ):
        """Offset parallel edges along the same path."""
        ntot = len(indices) + len(indices_inv)

        # This is straight so two vertices anyway
        # NOTE: all paths will be the same, which is why we need to offset them
        vs, ve = trans(paths[indices[0]].vertices)

        # Move orthogonal to the line
        fracs = (
            (vs - ve) / np.sqrt(((vs - ve) ** 2).sum()) @ np.array([[0, 1], [-1, 0]])
        )

        # NOTE: for now treat both direction the same
        for i, idx in enumerate(indices + indices_inv):
            # Offset the path
            paths[idx].vertices = trans_inv(
                trans(paths[idx].vertices) + fracs * offset * (i - ntot / 2)
            )

    def _compute_loop_path(
        self,
        vcoord_fig,
        vpath,
        vsize,
        angle1,
        angle2,
        trans_inv,
        looptension,
    ):
        # Shorten at starting angle
        start = self._get_shorter_edge_coords(vpath, vsize, angle1) + vcoord_fig
        # Shorten at end angle
        end = self._get_shorter_edge_coords(vpath, vsize, angle2) + vcoord_fig

        aux1 = (start - vcoord_fig) * looptension + vcoord_fig
        aux2 = (end - vcoord_fig) * looptension + vcoord_fig

        vertices = np.vstack(
            [
                start,
                aux1,
                aux2,
                end,
            ]
        )
        codes = ["MOVETO"] + ["CURVE4"] * 3

        # Offset to place and transform to data coordinates
        vertices = trans_inv(vertices)
        codes = [getattr(mpl.path.Path, x) for x in codes]
        path = mpl.path.Path(
            vertices,
            codes=codes,
        )
        return path

    def _compute_edge_path(
        self,
        *args,
        **kwargs,
    ):
        tension = kwargs.pop("tension", 0)
        waypoints = kwargs.pop("waypoints", "none")
        ports = kwargs.pop("ports", (None, None))

        if (waypoints != "none") and (tension != 0):
            raise ValueError("Waypoints not supported for curved edges.")

        if waypoints != "none":
            return self._compute_edge_path_waypoints(waypoints, *args, **kwargs)

        if tension == 0:
            return self._compute_edge_path_straight(*args, **kwargs)

        return self._compute_edge_path_curved(
            tension,
            *args,
            ports=ports,
            **kwargs,
        )

    def _compute_edge_path_waypoints(
        self,
        waypoints,
        vcoord_data,
        vpath_fig,
        vsize_fig,
        trans,
        trans_inv,
        points_per_curve=30,
        **kwargs,
    ):

        if waypoints in ("x0y1", "y0x1"):
            assert self._layout_coordinate_system == "cartesian"

            # Coordinates in figure (default) coords
            vcoord_fig = trans(vcoord_data)

            if waypoints == "x0y1":
                waypoint = np.array([vcoord_fig[0][0], vcoord_fig[1][1]])
            else:
                waypoint = np.array([vcoord_fig[1][0], vcoord_fig[0][1]])

            # Angles of the straight lines
            theta0 = atan2(*((waypoint - vcoord_fig[0])[::-1]))
            theta1 = atan2(*((waypoint - vcoord_fig[1])[::-1]))

            # Shorten at starting vertex
            vs = (
                self._get_shorter_edge_coords(vpath_fig[0], vsize_fig[0], theta0)
                + vcoord_fig[0]
            )

            # Shorten at end vertex
            ve = (
                self._get_shorter_edge_coords(vpath_fig[1], vsize_fig[1], theta1)
                + vcoord_fig[1]
            )

            points = [vs, waypoint, ve]
            codes = ["MOVETO", "LINETO", "LINETO"]
            angles = (theta0, theta1)
        elif waypoints == "r0a1":
            assert self._layout_coordinate_system == "polar"

            r0, alpha0 = vcoord_data[0]
            r1, alpha1 = vcoord_data[1]
            idx_inner = np.argmin([r0, r1])
            idx_outer = 1 - idx_inner
            alpha_outer = [alpha0, alpha1][idx_outer]

            # FIXME: this is aware of chirality as stored by the layout function
            betas = np.linspace(alpha0, alpha1, points_per_curve)
            waypoints = [r0, r1][idx_inner] * np.vstack(
                [np.cos(betas), np.sin(betas)]
            ).T
            endpoint = [r0, r1][idx_outer] * np.array(
                [np.cos(alpha_outer), np.sin(alpha_outer)]
            )
            points = np.array(list(waypoints) + [endpoint])
            points = trans(points)
            codes = ["MOVETO"] + ["LINETO"] * len(waypoints)
            # FIXME: same as previus comment
            angles = (alpha0 + pi / 2, alpha1)

        else:
            raise NotImplementedError(
                f"Edge shortening with waypoints not implemented yet: {waypoints}.",
            )

        path = mpl.path.Path(
            points,
            codes=[getattr(mpl.path.Path, x) for x in codes],
        )

        path.vertices = trans_inv(path.vertices)
        return path, angles

    def _compute_edge_path_straight(
        self,
        vcoord_data,
        vpath_fig,
        vsize_fig,
        trans,
        trans_inv,
        **kwargs,
    ):

        # Coordinates in figure (default) coords
        vcoord_fig = trans(vcoord_data)

        points = []

        # Angle of the straight line
        theta = atan2(*((vcoord_fig[1] - vcoord_fig[0])[::-1]))

        # Shorten at starting vertex
        vs = (
            self._get_shorter_edge_coords(vpath_fig[0], vsize_fig[0], theta)
            + vcoord_fig[0]
        )
        points.append(vs)

        # Shorten at end vertex
        ve = (
            self._get_shorter_edge_coords(vpath_fig[1], vsize_fig[1], theta + pi)
            + vcoord_fig[1]
        )
        points.append(ve)

        codes = ["MOVETO", "LINETO"]
        path = mpl.path.Path(
            points,
            codes=[getattr(mpl.path.Path, x) for x in codes],
        )
        path.vertices = trans_inv(path.vertices)
        return path, (theta, theta + np.pi)

    def _compute_edge_path_curved(
        self,
        tension,
        vcoord_data,
        vpath_fig,
        vsize_fig,
        trans,
        trans_inv,
        ports=(None, None),
    ):
        """Shorten the edge path along a cubic Bezier between the vertex centres.

        The most important part is that the derivative of the Bezier at the start
        and end point towards the vertex centres: people notice if they do not.
        """

        # Coordinates in figure (default) coords
        vcoord_fig = trans(vcoord_data)

        dv = vcoord_fig[1] - vcoord_fig[0]
        edge_straight_length = np.sqrt((dv**2).sum())

        auxs = [None, None]
        for i in range(2):
            if ports[i] is not None:
                der = _get_port_unit_vector(ports[i], trans_inv)
                auxs[i] = der * edge_straight_length * tension + vcoord_fig[i]

        # Both ports defined, just use them and hope for the best
        # Obviously, if the user specifies ports that make no sense,
        # this is going to be a (technically valid) mess.
        if all(aux is not None for aux in auxs):
            pass

        # If no ports are specified (the most common case), compute
        # the Bezier and shorten it
        elif all(aux is None for aux in auxs):
            # Put auxs along the way
            auxs = np.array(
                [
                    vcoord_fig[0] + 0.33 * dv,
                    vcoord_fig[1] - 0.33 * dv,
                ]
            )
            # Right rotation from the straight edge
            dv_rot = -0.1 * dv @ np.array([[0, 1], [-1, 0]])
            # Shift the auxs orthogonal to the straight edge
            auxs += dv_rot * tension

        # First port is defined
        elif (auxs[0] is not None) and (auxs[1] is None):
            auxs[1] = auxs[0]

        # Second port is defined
        else:
            auxs[0] = auxs[1]

        vs = [None, None]
        thetas = [None, None]
        for i in range(2):
            thetas[i] = atan2(*((auxs[i] - vcoord_fig[i])[::-1]))
            vs[i] = (
                self._get_shorter_edge_coords(vpath_fig[i], vsize_fig[i], thetas[i])
                + vcoord_fig[i]
            )

        path = {
            "vertices": [
                vs[0],
                auxs[0],
                auxs[1],
                vs[1],
            ],
            "codes": ["MOVETO"] + ["CURVE4"] * 3,
        }

        path = mpl.path.Path(
            path["vertices"],
            codes=[getattr(mpl.path.Path, x) for x in path["codes"]],
        )

        # Return to data transform
        path.vertices = trans_inv(path.vertices)
        return path, tuple(thetas)

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

    def _update_arrows(
        self,
        which: str = "end",
    ) -> None:
        """Extract the start and/or end angles of the paths to compute arrows.

        Parameters:
            which: Which end of the edge to put an arrow on. Currently only "end" is accepted.

        NOTE: This function does *not* update the arrow sizes/_transforms to the correct dpi scaling.
        That's ok since the correct dpi scaling is set whenever there is a different figure (before
        first draw) and whenever a draw is called.
        """
        if not hasattr(self, "_arrows"):
            return

        transform = self.get_transform()
        trans = transform.transform

        for i, epath in enumerate(self.get_paths()):
            # Offset the arrow to point to the end of the edge
            self._arrows._offsets[i] = epath.vertices[-1]

            # Rotate the arrow to point in the direction of the edge
            apath = self._arrows._paths[i]
            # NOTE: because the tip of the arrow is at (0, 0) in patch space,
            # in theory it will rotate around that point already
            v2 = trans(epath.vertices[-1])
            v1 = trans(epath.vertices[-2])
            dv = v2 - v1
            theta = atan2(*(dv[::-1]))
            theta_old = self._arrows._angles[i]
            dtheta = theta - theta_old
            mrot = np.array([[cos(dtheta), sin(dtheta)], [-sin(dtheta), cos(dtheta)]])
            apath.vertices = apath.vertices @ mrot
            self._arrows._angles[i] = theta

    @_stale_wrapper
    def draw(self, renderer):
        # Visibility affects the children too
        if not self.get_visible():
            return

        self._update_paths()
        # This sets the arrow offsets
        self._update_children()

        super().draw(renderer)
        for child in self.get_children():
            # This sets the arrow sizes with dpi scaling
            child.draw(renderer)

    @property
    def stale(self):
        return super().stale

    @stale.setter
    def stale(self, val):
        mpl.collections.PatchCollection.stale.fset(self, val)
        if val and hasattr(self, "stale_callback_post"):
            self.stale_callback_post(self)

    @staticmethod
    def _compute_loops_per_angle(nloops, angles):
        if len(angles) == 0:
            return [(0, 2 * pi, nloops)]

        angles_sorted_closed = list(sorted(angles))
        angles_sorted_closed.append(angles_sorted_closed[0] + 2 * pi)
        deltas = np.diff(angles_sorted_closed)

        # Now we have the deltas and the total number of loops
        # 1. Assign all loops to the largest wedge
        idx_dmax = deltas.argmax()
        if nloops == 1:
            return [
                (
                    angles_sorted_closed[idx_dmax],
                    angles_sorted_closed[idx_dmax + 1],
                    nloops,
                )
            ]

        # 2. Check if any other wedges are larger than this
        # If not, we are done (this is the algo in igraph)
        dsplit = deltas[idx_dmax] / nloops
        if (deltas > dsplit).sum() < 2:
            return [
                (
                    angles_sorted_closed[idx_dmax],
                    angles_sorted_closed[idx_dmax + 1],
                    nloops,
                )
            ]

            # 3. Check how small the second-largest wedge would become
        idx_dsort = np.argsort(deltas)
        return [
            (
                angles_sorted_closed[idx_dmax],
                angles_sorted_closed[idx_dmax + 1],
                nloops - 1,
            ),
            (
                angles_sorted_closed[idx_dsort[-2]],
                angles_sorted_closed[idx_dsort[-2] + 1],
                1,
            ),
        ]

    @staticmethod
    def _get_shorter_edge_coords(vpath, vsize, theta):
        # Bound theta from -pi to pi (why is that not guaranteed?)
        theta = (theta + pi) % (2 * pi) - pi

        # Size zero vertices need no shortening
        if vsize == 0:
            return np.array([0, 0])

        for i in range(len(vpath)):
            v1 = vpath.vertices[i]
            v2 = vpath.vertices[(i + 1) % len(vpath)]
            theta1 = atan2(*((v1)[::-1]))
            theta2 = atan2(*((v2)[::-1]))

            # atan2 ranges ]-3.14, 3.14]
            # so it can be that theta1 is -3 and theta2 is +3
            # therefore we need two separate cases, one that cuts at pi and one at 0
            cond1 = theta1 <= theta <= theta2
            cond2 = (
                (theta1 + 2 * pi) % (2 * pi)
                <= (theta + 2 * pi) % (2 * pi)
                <= (theta2 + 2 * pi) % (2 * pi)
            )
            if cond1 or cond2:
                break
        else:
            raise ValueError("Angle for patch not found")

        # The edge meets the patch of the vertex on the v1-v2 size,
        # at angle theta from the center
        mtheta = tan(theta)
        if v2[0] == v1[0]:
            xe = v1[0]
        else:
            m12 = (v2[1] - v1[1]) / (v2[0] - v1[0])
            xe = (v1[1] - m12 * v1[0]) / (mtheta - m12)
        ye = mtheta * xe
        ve = np.array([xe, ye])
        return ve * vsize


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
        "cmap",
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
