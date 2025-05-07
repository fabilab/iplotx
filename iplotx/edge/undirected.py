from math import atan2, cos, pi, sin
import matplotlib as mpl


class UndirectedEdgeCollection(mpl.collections.PatchCollection):
    def __init__(self, *args, **kwargs):
        kwargs["match_original"] = True
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

        visual_vertices = self._visual_vertices
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
        loop_vertex_dict = {}
        for i, edge_vertices in enumerate(visual_vertices):
            if edge_vertices[0] == edge_vertices[1]:
                if edge_vertices[0] not in loop_vertex_dict:
                    loop_vertex_dict[edge_vertices[0]] = {
                        "indices": [],
                        "sizes": [],
                        "edge_angles": [],
                    }
                    if self._directed:
                        loop_vertex_dict[edge_vertices[0]]["arrow_sizes"] = []
                        loop_vertex_dict[edge_vertices[0]]["arrow_widths"] = []
                loop_vertex_dict[edge_vertices[0]]["indices"].append(i)

        # Get actual coordinates of the vertex border (rough)
        paths = []
        for i, edge_vertices in enumerate(visual_vertices):
            if self._directed:
                if (self._arrow_sizes is None) or (self._arrow_widths is None):
                    arrow_size = 0
                    arrow_width = 0
                else:
                    arrow_size = self._arrow_sizes[i]
                    arrow_width = self._arrow_widths[i]

            # Loops are positioned post-facto in the space left by the othter edges
            if edge_vertices[0] == edge_vertices[1]:
                paths.append(None)
                loop_vertex_dict[edge_vertices[0]]["sizes"].append(
                    self._loop_sizes[i],
                )
                if self._directed:
                    loop_vertex_dict[edge_vertices[0]]["arrow_sizes"].append(
                        arrow_size,
                    )
                    loop_vertex_dict[edge_vertices[0]]["arrow_widths"].append(
                        arrow_width,
                    )
                continue

            coords = np.vstack(
                [
                    edge_vertices[0].position,
                    edge_vertices[1].position,
                ]
            )
            coordst = trans(coords)
            sizes = self._get_edge_vertex_sizes(edge_vertices)
            if self._style is not None and self._style.get("curved", False):
                curved = True
            else:
                curved = False

            path = self._compute_path_undirected(
                coordst,
                sizes,
                trans_inv,
                curved,
            )

            # Collect angles for this vertex, to be used for loops plotting below
            angles = self._compute_edge_angles(path, trans, self._directed, curved)
            if edge_vertices[0] in loop_vertex_dict:
                loop_vertex_dict[edge_vertices[0]]["edge_angles"].append(angles[0])
            if edge_vertices[1] in loop_vertex_dict:
                loop_vertex_dict[edge_vertices[1]]["edge_angles"].append(angles[1])

            # Add the path for this non-loop edge
            paths.append(path)

        # Deal with loops at the end
        for visual_vertex, ldict in loop_vertex_dict.items():
            coords = np.vstack([visual_vertex.position] * 2)
            coordst = trans(coords)
            vertex_size = self._get_edge_vertex_sizes([visual_vertex])[0]

            edge_angles = ldict["edge_angles"]
            if edge_angles:
                edge_angles.sort()
                # Circle around
                edge_angles.append(edge_angles[0] + 2 * pi)
                wedges = [
                    (a2 - a1) for a1, a2 in zip(edge_angles[:-1], edge_angles[1:])
                ]
                # Argsort
                imax = max(range(len(wedges)), key=lambda i: wedges[i])
                angle1, angle2 = edge_angles[imax], edge_angles[imax + 1]
            else:
                # Isolated vertices with loops
                angle1, angle2 = -pi, pi

            nloops = len(ldict["indices"])
            for i in range(nloops):
                angle1i = angle1 + (angle2 - angle1) * i / nloops
                angle2i = angle1 + (angle2 - angle1) * (i + 1) / nloops
                if self._directed:
                    loop_kwargs = {
                        "arrow_size": ldict["arrow_sizes"][i],
                        "arrow_width": ldict["arrow_widths"][i],
                    }
                else:
                    loop_kwargs = {}
                path = self._compute_loop_path(
                    coordst[0],
                    vertex_size,
                    ldict["sizes"][i],
                    angle1i,
                    angle2i,
                    trans_inv,
                    angle_padding_fraction=0.1,
                    **loop_kwargs,
                )
                paths[ldict["indices"][i]] = path

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

    def _compute_path_undirected(self, coordst, sizes, trans_inv, curved):
        path = {"vertices": [], "codes": []}
        path["codes"].append("MOVETO")
        if not curved:
            path["codes"].append("LINETO")

            # Start
            theta = atan2(*((coordst[1] - coordst[0])[::-1]))
            voff = 0 * coordst[0]
            voff[:] = [cos(theta), sin(theta)]
            voff *= sizes[0] / 2
            path["vertices"].append(coordst[0] + voff)

            # End
            voff[:] = [cos(theta), sin(theta)]
            voff *= sizes[1] / 2
            path["vertices"].append(coordst[1] - voff)
        else:
            path["codes"].extend(["CURVE4"] * 3)

            aux1, aux2 = get_bezier_control_points_for_curved_edge(
                *coordst.ravel(),
                curved,
            )

            # Start
            theta = atan2(*((aux1 - coordst[0])[::-1]))
            voff = 0 * coordst[0]
            voff[:] = [cos(theta), sin(theta)]
            voff *= sizes[0] / 2
            path["vertices"].append(coordst[0] + voff)

            # Bezier
            path["vertices"].append(aux1)
            path["vertices"].append(aux2)

            # End
            theta = atan2(*((coordst[1] - aux2)[::-1]))
            voff = 0 * coordst[0]
            voff[:] = [cos(theta), sin(theta)]
            voff *= sizes[1] / 2
            path["vertices"].append(coordst[1] - voff)

            # This is a dirty trick to make the facecolor work
            # without making a separate Patch, which would be a little messy
            path["codes"].extend(["CURVE4"] * 3)
            path["vertices"].extend(path["vertices"][-2::-1])

        path = mpl.path.Path(
            path["vertices"],
            codes=[getattr(mpl.path.Path, x) for x in path["codes"]],
        )
        path.vertices = trans_inv(path.vertices)
        return path

    def draw(self, renderer):
        if self._vertex_patches is not None:
            self._paths = self._compute_paths()
        return super().draw(renderer)

    @property
    def stale(self):
        return super().stale

    @stale.setter
    def stale(self, val):
        PatchCollection.stale.fset(self, val)
        if val and hasattr(self, "stale_callback_post"):
            self.stale_callback_post(self)


def make_stub_patch(**kwargs):
    """Make a stub undirected edge patch, without actual path information."""
    kwargs['clip_on'] = kwargs.get('clip_on', True)

    # NOTE: the path is overwritten later anyway, so no reason to spend any time here
    art = mpl.patches.PathPatch(
        mpl.path.Path([[0, 0]]),
        **kwargs,
    )
