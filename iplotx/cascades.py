from typing import (
    Any,
)
import numpy as np
import pandas as pd

from .typing import (
    TreeType,
)
from .ingest.typing import (
    TreeDataProvider,
)
import matplotlib as mpl

from .style import (
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
    ):
        style = copy_with_deep_values(style)

        # NOTE: there is a weird bug in pandas when using generic Hashable-s
        # with .loc. Seems like doing .T[...] works for individual index
        # elements only though
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
        for prop in ("facecolor", "edgecolor"):
            if prop in style:
                nodes_unordered |= set(style[prop].keys())

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

        if layout_name == "horizontal":
            if orientation == "right":
                depth_max = layout.values[:, 0].max()
            else:
                depth_max = layout.values[:, 0].min()
        elif layout_name == "vertical":
            if orientation == "descending":
                depth_max = layout.values[:, 1].min()
            else:
                depth_max = layout.values[:, 1].max()
        elif layout_name == "radial":
            # layout values are: r, theta
            depth_max = layout.values[:, 0].max()
            nleaves = len(provider(tree).get_leaves())

        cascading_patches = []
        for node in drawing_order:
            stylei = rotate_style(style, key=node)
            stylei.pop("extend", None)
            # Default alpha is 0.5 for simple colors
            if isinstance(stylei.get("facecolor", None), str) and (
                "alpha" not in stylei
            ):
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
                        xright = depth_max if extend else leaves_coords[:, 0].max()
                    else:
                        xleft = depth_max if extend else leaves_coords[:, 0].min()
                        xright = node_coords[0] + bl
                elif layout_name == "vertical":
                    xleft = leaves_coords[:, 0].min() - 0.5
                    xright = leaves_coords[:, 0].max() + 0.5
                    if orientation == "descending":
                        ytop = node_coords[1] + bl
                        ybot = depth_max if extend else leaves_coords[:, 1].min()
                    else:
                        ytop = depth_max if extend else leaves_coords[:, 1].max()
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
                rmax = depth_max if extend else leaves_coords[:, 0].max()
                thetamin = leaves_coords[:, 1].min() - 0.5 * dtheta
                thetamax = leaves_coords[:, 1].max() + 0.5 * dtheta
                thetas = np.linspace(
                    thetamin, thetamax, max(30, (thetamax - thetamin) // 3)
                )
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
        )
