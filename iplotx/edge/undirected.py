from math import atan2, tan, cos, pi, sin
import numpy as np
import matplotlib as mpl


class UndirectedEdgeCollection(mpl.collections.PatchCollection):
    def __init__(self, *args, **kwargs):
        kwargs["match_original"] = True
        self._vertex_ids = kwargs.pop("vertex_ids", None)
        self._vertex_centers = kwargs.pop("vertex_centers", None)
        self._vertex_paths = kwargs.pop("vertex_paths", None)
        self._style = kwargs.pop("style", None)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_edge_vertex_sizes(edge_vertices):
        sizes = []
        for visual_vertex in edge_vertices:
            if visual_vertex.size is not None:
                sizes.append(visual_vertex.size)
            else:
                sizes.append(max(visual_vertex.width, visual_vertex.height))
        return sizes

    @staticmethod
    def _compute_edge_angles(path, trans, directed, curved):
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

    def _compute_paths(self, transform=None):
        import numpy as np

        vids = self._vertex_ids
        vpaths = self._vertex_paths
        vcenters = self._vertex_centers
        if transform is None:
            transform = self.get_transform()
        trans = transform.transform
        trans_inv = transform.inverted().transform

        # Loops split the largest wedge left open by other
        # edges of that vertex. The algo is:
        # (i) Find what vertices each loop belongs to
        # (ii) While going through the edges, record the angles
        #      for vertices with loops
        # (iii) Plot each loop based on the recorded angles
        # NOTE: one could improve this based on greediness

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

            # Shorten edge
            if not self._style.get("curved", False):
                path = self._shorten_path_undirected_straight(
                    vcoord_fig,
                    vpath_fig,
                    trans_inv,
                )
            else:
                raise NotImplementedError("Curved edges not implemented yet.")

            # Collect angles for this vertex, to be used for loops plotting below
            if (v1 in loop_vertex_dict) or (v2 in loop_vertex_dict):
                angles = self._compute_edge_angles(
                    path,
                    trans,
                    self._style.get("curved", False),
                )
                if v1 in loop_vertex_dict:
                    loop_vertex_dict[v1]["edge_angles"].append(angles[0])
                if v2 in loop_vertex_dict:
                    loop_vertex_dict[v2]["edge_angles"].append(angles[1])

            # Add the path for this non-loop edge
            paths.append(path)

        # TODO: uncomment
        # 3. Deal with loops at the end
        # for visual_vertex, ldict in loop_vertex_dict.items():
        #    coords = np.vstack([visual_vertex.position] * 2)
        #    coordst = trans(coords)
        #    vertex_size = self._get_edge_vertex_sizes([visual_vertex])[0]

        #    edge_angles = ldict["edge_angles"]
        #    if edge_angles:
        #        edge_angles.sort()
        #        # Circle around
        #        edge_angles.append(edge_angles[0] + 2 * pi)
        #        wedges = [
        #            (a2 - a1) for a1, a2 in zip(edge_angles[:-1], edge_angles[1:])
        #        ]
        #        # Argsort
        #        imax = max(range(len(wedges)), key=lambda i: wedges[i])
        #        angle1, angle2 = edge_angles[imax], edge_angles[imax + 1]
        #    else:
        #        # Isolated vertices with loops
        #        angle1, angle2 = -pi, pi

        #    nloops = len(ldict["indices"])
        #    for i in range(nloops):
        #        angle1i = angle1 + (angle2 - angle1) * i / nloops
        #        angle2i = angle1 + (angle2 - angle1) * (i + 1) / nloops
        #        if self._directed:
        #            loop_kwargs = {
        #                "arrow_size": ldict["arrow_sizes"][i],
        #                "arrow_width": ldict["arrow_widths"][i],
        #            }
        #        else:
        #            loop_kwargs = {}
        #        path = self._compute_loop_path(
        #            coordst[0],
        #            vertex_size,
        #            ldict["sizes"][i],
        #            angle1i,
        #            angle2i,
        #            trans_inv,
        #            angle_padding_fraction=0.1,
        #            **loop_kwargs,
        #        )
        #        paths[ldict["indices"][i]] = path

        return paths

    def _compute_loop_path(
        self,
        coordt,
        vertex_size,
        loop_size,
        angle1,
        angle2,
        trans_inv,
        angle_padding_fraction=0.1,
    ):
        import numpy as np

        # Special argument for loop size to scale with vertices
        if loop_size < 0:
            loop_size = -loop_size * vertex_size

        # Pad angles to make a little space for tight arrowheads
        angle1, angle2 = (
            angle1 * (1 - angle_padding_fraction) + angle2 * angle_padding_fraction,
            angle1 * angle_padding_fraction + angle2 * (1 - angle_padding_fraction),
        )

        # Too large wedges, use a quarter as the largest loop wedge
        if angle2 - angle1 > pi / 3:
            angle_mid = (angle2 + angle1) * 0.5
            angle1 = angle_mid - pi / 6
            angle2 = angle_mid + pi / 6

        start = vertex_size / 2 * np.array([cos(angle1), sin(angle1)])
        end = vertex_size / 2 * np.array([cos(angle2), sin(angle2)])
        amix = 0.05
        aux1 = loop_size * np.array(
            [
                cos(angle1 * (1 - amix) + angle2 * amix),
                sin(angle1 * (1 - amix) + angle2 * amix),
            ]
        )
        aux2 = loop_size * np.array(
            [
                cos(angle1 * amix + angle2 * (1 - amix)),
                sin(angle1 * amix + angle2 * (1 - amix)),
            ]
        )

        vertices = np.vstack(
            [
                start,
                aux1,
                aux2,
                end,
                aux2,
                aux1,
                start,
            ]
        )
        codes = ["MOVETO"] + ["CURVE4"] * 6

        # Offset to place and transform to data coordinates
        vertices = trans_inv(coordt + vertices)
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
        vs = _get_shorter_edge_coords(vpath_fig[0], theta) + vcoord_fig[0]
        path["vertices"].append(vs)

        # Shorten at end vertex
        ve = _get_shorter_edge_coords(vpath_fig[1], theta + pi) + vcoord_fig[1]
        path["vertices"].append(ve)

        path = mpl.path.Path(
            path["vertices"],
            codes=[getattr(mpl.path.Path, x) for x in path["codes"]],
        )
        path.vertices = trans_inv(path.vertices)
        return path

    def draw(self, renderer):
        if self._vertex_paths is not None:
            self._paths = self._compute_paths()
        return super().draw(renderer)

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

    # NOTE: the path is overwritten later anyway, so no reason to spend any time here
    art = mpl.patches.PathPatch(
        mpl.path.Path([[0, 0]]),
        **kwargs,
    )
    return art


def _get_shorter_edge_coords(vpath, theta):
    for i in range(len(vpath)):
        v1 = vpath.vertices[i]
        v2 = vpath.vertices[(i + 1) % len(vpath)]
        theta1 = atan2(*((v1)[::-1]))
        theta2 = atan2(*((v2)[::-1]))
        # atan2 ranges ]-3.14, 3.14]
        if theta1 <= theta <= theta2:
            break
        if (
            (theta1 + pi) % (2 * pi)
            <= (theta + pi) % (2 * pi)
            <= (theta2 + pi) % (2 * pi)
        ):
            break
    else:
        raise ValueError("Angle for patch not found")

    # The edge meets the patch of the vertex on the v1-v2 size, at angle theta from the center
    m12 = (v2[1] - v1[1]) / (v2[0] - v1[0])
    mtheta = tan(theta)
    xe = (v1[1] - m12 * v1[0]) / (mtheta - m12)
    ye = mtheta * xe
    ve = np.array([xe, ye])
    return ve
