from typing import (
    Any,
    Optional,
)
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl

from ..typing import (
    TreeType,
)
from ..ingest.typing import (
    TreeDataProvider,
)
from ..style import (
    copy_with_deep_values,
    rotate_style,
)


class CascadeCollection(mpl.collections.PatchCollection):
    def __init__(
        self,
        tree: TreeType,
        layout: pd.DataFrame,
        layout_name: str,
        orientation: str,
        style: dict[str, Any],
        provider: TreeDataProvider,
        transform: mpl.transforms.Transform,
        maxdepth: Optional[float] = None,
    ):
        self._layout_name = layout_name
        self._orientation = orientation
        style = copy_with_deep_values(style)
        zorder = style.get("zorder", 0)

        # NOTE: there is a weird bug in pandas when using generic Hashable-s
        # with .loc. Seems like doing .T[...] works for individual index
        # elements only though (i.e. using __getitem__ a la dict)
        def get_node_coords(node):
            return layout.T[node].values

        def get_leaves_coords(leaves):
            return np.array(
                [get_node_coords(leaf) for leaf in leaves],
            )

        if "color" in style:
            style["facecolor"] = style["edgecolor"] = style.pop("color")
        extend = style.get("extend", False)

        # These patches need at least a facecolor (usually) or an edgecolor
        # so it's safe to make a list from these
        nodes_unordered = set()
        for prop in ("facecolor", "edgecolor", "linewidth", "linestyle"):
            if prop in style:
                value = style[prop]
                if isinstance(value, dict):
                    nodes_unordered |= set(value.keys())

        if len(nodes_unordered) == 0:
            warnings.warn(
                "No nodes found in the style for the cascading patches. "
                "Please provide a style with at least one dict-like "
                "specification among the following properties: 'facecolor', "
                "'edgecolor', 'color', 'linewidth', or 'linestyle'.",
            )

        # Draw the patches from the closest to the root (earlier drawing)
        # to the closer to the leaves (later drawing).
        drawing_order = []
        for node in provider(tree).preorder():
            if node in nodes_unordered:
                drawing_order.append(node)

        if layout_name not in ("horizontal", "vertical", "radial"):
            raise NotImplementedError(
                f"Cascading patches not implemented for layout: {layout_name}.",
            )

        self._maxdepth = maxdepth

        cascading_patches = []
        nleaves = sum(1 for leaf in provider(tree).get_leaves())
        for node in drawing_order:
            stylei = rotate_style(style, key=node)
            stylei.pop("extend", None)
            # Default alpha is 0.5 for simple colors
            if isinstance(stylei.get("facecolor", None), str) and ("alpha" not in stylei):
                stylei["alpha"] = 0.5

            provider_node = provider(node)
            bl = provider_node.get_branch_length_default_to_one(node)
            node_coords = get_node_coords(node).copy()
            leaves_coords = get_leaves_coords(provider_node.get_leaves())
            if len(leaves_coords) == 0:
                leaves_coords = np.array([node_coords])

            if layout_name in ("horizontal", "vertical"):
                if layout_name == "horizontal":
                    ybot = leaves_coords[:, 1].min() - 0.5
                    ytop = leaves_coords[:, 1].max() + 0.5
                    if orientation == "right":
                        xleft = node_coords[0] - bl
                        xright = maxdepth if extend else leaves_coords[:, 0].max()
                    else:
                        xleft = maxdepth if extend else leaves_coords[:, 0].min()
                        xright = node_coords[0] + bl
                elif layout_name == "vertical":
                    xleft = leaves_coords[:, 0].min() - 0.5
                    xright = leaves_coords[:, 0].max() + 0.5
                    if orientation == "descending":
                        ytop = node_coords[1] + bl
                        ybot = maxdepth if extend else leaves_coords[:, 1].min()
                    else:
                        ytop = maxdepth if extend else leaves_coords[:, 1].max()
                        ybot = node_coords[1] - bl

                patch = mpl.patches.Rectangle(
                    (xleft, ybot),
                    xright - xleft,
                    ytop - ybot,
                    **stylei,
                )
            elif layout_name == "radial":
                dtheta = 2 * np.pi / nleaves
                rmin = node_coords[0] - bl
                rmax = maxdepth if extend else leaves_coords[:, 0].max()
                thetamin = leaves_coords[:, 1].min() - 0.5 * dtheta
                thetamax = leaves_coords[:, 1].max() + 0.5 * dtheta
                thetas = np.linspace(thetamin, thetamax, max(30, (thetamax - thetamin) // 3))
                xs = list(rmin * np.cos(thetas)) + list(rmax * np.cos(thetas[::-1]))
                ys = list(rmin * np.sin(thetas)) + list(rmax * np.sin(thetas[::-1]))
                points = list(zip(xs, ys))
                points.append(points[0])
                codes = ["MOVETO"] + ["LINETO"] * (len(points) - 2) + ["CLOSEPOLY"]

                if "edgecolor" not in stylei:
                    stylei["edgecolor"] = "none"

                path = mpl.path.Path(
                    points,
                    codes=[getattr(mpl.path.Path, code) for code in codes],
                )
                patch = mpl.patches.PathPatch(
                    path,
                    **stylei,
                )

            cascading_patches.append(patch)

        super().__init__(
            cascading_patches,
            transform=transform,
            match_original=True,
            zorder=zorder,
        )

    def get_maxdepth(self) -> float:
        """Get the maxdepth of the cascades.

        Returns: The maximum depth of the cascading patches.
        """
        return self._maxdepth

    def set_maxdepth(self, maxdepth: float):
        """Set the maximum depth of the cascading patches.

        Parameters:
            maxdepth: The new maximum depth for the cascades.

        NOTE: Calling this function updates the cascade patches
        without chechking whether the extent style requires it.
        """
        self._maxdepth = maxdepth
        self._update_maxdepth()

    def _update_maxdepth(self):
        """Update the cascades with a new max depth.

        Note: This function changes the paths without checking whether
        the extent is set or not.
        """
        layout_name = self._layout_name
        orientation = self._orientation

        # This being a PatchCollection, we have to touch the paths
        if layout_name == "radial":
            for path in self.get_paths():
                # Old radii
                r2old = np.linalg.norm(path.vertices[-2])
                # Update the outer part of the wedge patch
                path.vertices[(len(path.vertices) - 1) // 2 :] *= self.get_maxdepth() / r2old
            return

        if (layout_name, orientation) == ("horizontal", "right"):
            for path in self.get_paths():
                path.vertices[[1, 2], 0] = self.get_maxdepth()
        elif (layout_name, orientation) == ("horizontal", "left"):
            for path in self.get_paths():
                path.vertices[[0, 3], 0] = self.get_maxdepth()
        elif (layout_name, orientation) == ("vertical", "descending"):
            for path in self.get_paths():
                path.vertices[[0, 1], 1] = self.get_maxdepth()
        elif (layout_name, orientation) == ("vertical", "ascending"):
            for path in self.get_paths():
                path.vertices[[2, 3], 1] = self.get_maxdepth()
        else:
            raise ValueError(
                f"Layout name and orientation not supported: {layout_name}, {orientation}."
            )
