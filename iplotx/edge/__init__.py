from math import atan2, tan, cos, pi, sin
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib as mpl

from .common import _compute_loops_per_angle
from ..utils.matplotlib import (
    _compute_mid_coord_and_rot,
    _stale_wrapper,
    _forwarder,
)
from ..style import (
    rotate_style,
)
from ..label import LabelCollection
from .arrow import EdgeArrowCollection


@_forwarder(
    (
        "set_clip_path",
        "set_clip_box",
        "set_snap",
        "set_sketch_params",
        "set_figure",
        "set_animated",
        "set_picker",
    )
)
class EdgeCollection(mpl.collections.PatchCollection):
    def __init__(self, *args, **kwargs):
        kwargs["match_original"] = True
        self._vertex_ids = kwargs.pop("vertex_ids", None)
        self._vertex_collection = kwargs.pop("vertex_collection", None)
        self._style = kwargs.pop("style", None)
        self._labels = kwargs.pop("labels", None)
        self.directed = kwargs.pop("directed", False)
        if "cmap" in self._style:
            kwargs["cmap"] = self._style["cmap"]
            kwargs["norm"] = self._style["norm"]
        super().__init__(*args, **kwargs)

        if self.directed:
            self._arrows = EdgeArrowCollection(self)
        else:
            self._arrows = None
        self._processed = False

    def set_array(self, array):
        """Set the array for cmap/norm coloring, but keep the facecolors as set (usually 'none')."""
        super().set_array(array)
        if self._arrows is not None:
            self._arrows.set_array(array)

    def get_labels(self):
        if hasattr(self, "_label_collection"):
            return self._label_collection
        else:
            return None

    def get_mappable(self):
        """Return mappable for colorbar."""
        return self

    @staticmethod
    def _compute_edge_angles(path, size, trans):
        """Compute edge angles for both starting and ending vertices.

        NOTE: The domain of atan2 is (-pi, pi].
        """
        positions = trans(path.vertices)

        # first angle
        x1, y1 = positions[0]
        x2, y2 = positions[1]
        angle1 = atan2(y2 - y1, x2 - x1)

        # second angle
        x1, y1 = positions[-1]
        x2, y2 = positions[-2]
        angle2 = atan2(y2 - y1, x2 - x1)
        return (angle1, angle2)

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
            offset1 = self._vertex_collection.get_offsets()[index[v1]]
            offset2 = self._vertex_collection.get_offsets()[index[v2]]
            voffsets.append((offset1, offset2))

            path1 = self._vertex_collection.get_paths()[index[v1]]
            path2 = self._vertex_collection.get_paths()[index[v1]]
            vpaths.append((path1, path2))

            # NOTE: This needs to be computed here because the
            # VertexCollection._transforms are reset each draw in order to
            # accomodate for DPI changes on the canvas
            size1 = self._vertex_collection.get_sizes_dpi()[index[v1]]
            size2 = self._vertex_collection.get_sizes_dpi()[index[v1]]
            vsizes.append((size1, size2))

        return {
            "ids": self._vertex_ids,
            "offsets": voffsets,
            "paths": vpaths,
            "sizes": vsizes,
        }

    def _compute_paths(self, transform=None):
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

        if transform is None:
            transform = self.get_transform()
        trans = transform.transform
        trans_inv = transform.inverted().transform

        # 1. Make a list of vertices with loops, and store them for later
        loop_vertex_dict = {}
        for i, (v1, v2) in enumerate(vids):
            if v1 != v2:
                continue
            if v1 not in loop_vertex_dict:
                loop_vertex_dict[v1] = {
                    "indices": [],
                    "edge_angles": [],
                }
            loop_vertex_dict[v1]["indices"].append(i)

        # 2. Make paths for non-loop edges
        # NOTE: keep track of parallel edges to offset them
        parallel_edges = defaultdict(list)

        # Get actual coordinates of the vertex border
        paths = []
        for i, (v1, v2) in enumerate(vids):
            # Postpone loops (step 3)
            if v1 == v2:
                paths.append(None)
                continue

            # Coordinates of the adjacent vertices, in data coords
            vcoord_data = vcenters[i]

            # Coordinates in figure (default) coords
            vcoord_fig = trans(vcoord_data)

            # Vertex paths in figure (default) coords
            vpath_fig = vpaths[i]

            # Vertex size
            vsize_fig = vsizes[i]

            # Shorten edge
            if not self._style.get("curved", False):
                path = self._shorten_path_undirected_straight(
                    vcoord_fig,
                    vpath_fig,
                    vsize_fig,
                    trans_inv,
                )
            else:
                edge_stylei = rotate_style(self._style, index=i, id=(v1, v2))
                tension = edge_stylei.get("tension", 1.5)
                path = self._shorten_path_undirected_curved(
                    vcoord_fig,
                    vpath_fig,
                    vsize_fig,
                    trans_inv,
                    tension,
                )

            # Collect angles for this vertex, to be used for loops plotting below
            if (v1 in loop_vertex_dict) or (v2 in loop_vertex_dict):
                angles = self._compute_edge_angles(
                    path,
                    vsize_fig,
                    trans,
                )
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
                nparallel = len(indices)
                nparallel_inv = len(indices_inv)
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
            nloops_per_angle = _compute_loops_per_angle(nloops, edge_angles)

            idx = 0
            for theta1, theta2, nloops in nloops_per_angle:
                # Angular size of each loop in this wedge
                delta = (theta2 - theta1) / nloops

                # Iterate over individual loops
                for j in range(nloops):
                    thetaj1 = theta1 + j * delta
                    # Use 60 degrees as the largest possible loop wedge
                    thetaj2 = thetaj1 + min(delta, pi / 3)

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

        return paths

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
        start = _get_shorter_edge_coords(vpath, vsize, angle1) + vcoord_fig
        # Shorten at end angle
        end = _get_shorter_edge_coords(vpath, vsize, angle2) + vcoord_fig

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

    def _shorten_path_undirected_straight(
        self,
        vcoord_fig,
        vpath_fig,
        vsize_fig,
        trans_inv,
    ):
        # Straight SVG instructions
        path = {
            "vertices": [],
            "codes": ["MOVETO", "LINETO"],
        }

        # Angle of the straight line
        theta = atan2(*((vcoord_fig[1] - vcoord_fig[0])[::-1]))

        # Shorten at starting vertex
        vs = _get_shorter_edge_coords(vpath_fig[0], vsize_fig[0], theta) + vcoord_fig[0]
        path["vertices"].append(vs)

        # Shorten at end vertex
        ve = (
            _get_shorter_edge_coords(vpath_fig[1], vsize_fig[1], theta + pi)
            + vcoord_fig[1]
        )
        path["vertices"].append(ve)

        path = mpl.path.Path(
            path["vertices"],
            codes=[getattr(mpl.path.Path, x) for x in path["codes"]],
        )
        path.vertices = trans_inv(path.vertices)
        return path

    def _shorten_path_undirected_curved(
        self,
        vcoord_fig,
        vpath_fig,
        vsize_fig,
        trans_inv,
        tension,
    ):
        # Angle of the straight line
        theta = atan2(*((vcoord_fig[1] - vcoord_fig[0])[::-1]))

        # Shorten at starting vertex
        vs = _get_shorter_edge_coords(vpath_fig[0], vsize_fig, theta) + vcoord_fig[0]

        # Shorten at end vertex
        ve = (
            _get_shorter_edge_coords(vpath_fig[1], vsize_fig, theta + pi)
            + vcoord_fig[1]
        )

        edge_straight_length = np.sqrt(((ve - vs) ** 2).sum())

        aux1 = vs + 0.33 * (ve - vs)
        aux2 = vs + 0.67 * (ve - vs)

        # Move Bezier points orthogonal to the line
        fracs = (
            (vs - ve) / np.sqrt(((vs - ve) ** 2).sum()) @ np.array([[0, 1], [-1, 0]])
        )
        aux1 += 0.1 * fracs * tension * edge_straight_length
        aux2 += 0.1 * fracs * tension * edge_straight_length

        path = {
            "vertices": [
                vs,
                aux1,
                aux2,
                ve,
            ],
            "codes": ["MOVETO"] + ["CURVE4"] * 3,
        }

        path = mpl.path.Path(
            path["vertices"],
            codes=[getattr(mpl.path.Path, x) for x in path["codes"]],
        )
        path.vertices = trans_inv(path.vertices)
        return path

    def _compute_labels(self):
        transform = self.get_transform()
        trans = transform.transform

        style = self._style.get("label", None) if self._style is not None else {}
        offsets = []
        if not style.get("rotate", True):
            rotations = []
        for path in self._paths:
            offset, rotation = _compute_mid_coord_and_rot(path, trans)
            offsets.append(offset)
            if not style.get("rotate", True):
                rotations.append(rotation)

        if not hasattr(self, "_label_collection"):
            self._label_collection = LabelCollection(
                self._labels,
                style=style,
                offsets=offsets,
                transform=self.axes.transData,
            )

            # Forward a bunch of mpl settings that are needed
            self._label_collection.set_figure(self.figure)
            self._label_collection.axes = self.axes
            # forward the clippath/box to the children need this logic
            # because mpl exposes some fast-path logic
            clip_path = self.get_clip_path()
            if clip_path is None:
                clip_box = self.get_clip_box()
                self._label_collection.set_clip_box(clip_box)
            else:
                self._label_collection.set_clip_path(clip_path)

            # Finally make the patches
            self._label_collection._create_artists()

        if not style.get("rotate", True):
            self._label_collection.set_rotations(rotations)

    def _process(self):
        # Forward mpl properties to children
        # TODO sort out all of the things that need to be forwarded
        for child in self.get_children():
            # set the figure & axes on child, this ensures each artist
            # down the hierarchy knows where to draw
            if hasattr(child, "set_figure"):
                child.set_figure(self.figure)
            child.axes = self.axes

            # forward the clippath/box to the children need this logic
            # because mpl exposes some fast-path logic
            clip_path = self.get_clip_path()
            if clip_path is None:
                clip_box = self.get_clip_box()
                child.set_clip_box(clip_box)
            else:
                child.set_clip_path(clip_path)

        self._processed = True

    def _set_edge_info_for_arrows(
        self,
        which="end",
        transform=None,
    ):
        """Extract the start and/or end angles of the paths to compute arrows."""
        if transform is None:
            transform = self.get_transform()
        trans = transform.transform
        trans_inv = transform.inverted().transform

        arrow_offsets = self._arrows._offsets
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

    def get_children(self):
        children = []
        if self._arrows is not None:
            children.append(self._arrows)
        if hasattr(self, "_label_collection"):
            children.append(self._label_collection)
        return children

    @_stale_wrapper
    def draw(self, renderer, *args, **kwds):
        # Visibility of the edges affects also the children
        if not self.get_visible():
            return

        if not self._processed:
            self._process()

        if self._vertex_collection is not None:
            self._paths = self._compute_paths()
            if self._labels is not None:
                self._compute_labels()

        super().draw(renderer)
        if self._arrows is not None:
            self._set_edge_info_for_arrows(which="end")

        for child in self.get_children():
            child.draw(renderer, *args, **kwds)

    @property
    def stale(self):
        return super().stale

    @stale.setter
    def stale(self, val):
        mpl.collections.PatchCollection.stale.fset(self, val)
        if val and hasattr(self, "stale_callback_post"):
            self.stale_callback_post(self)


def make_stub_patch(**kwargs):
    """Make a stub undirected edge patch, without actual path information."""
    kwargs["clip_on"] = kwargs.get("clip_on", True)
    if ("color" in kwargs) and ("edgecolor" not in kwargs):
        kwargs["edgecolor"] = kwargs.pop("color")
    # Edges are always hollow, because they are not closed paths
    kwargs["facecolor"] = "none"

    # Forget specific properties that are not supported here
    forbidden_props = [
        "curved",
        "tension",
        "looptension",
        "offset",
        "label",
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
