"""
All artists defined in iplotx.
"""

from .network import NetworkArtist
from .network.groups import GroupingCollection
from .vertex import VertexCollection
from .edge import EdgeCollection
from .label import LabelCollection
from .edge.arrow import EdgeArrowCollection
from .edge.leaf import LeafEdgeCollection
from .art3d.vertex import Vertex3DCollection
from .art3d.edge import Edge3DCollection
from .tree import TreeArtist
from .tree.cascades import CascadeCollection


___all__ = (
    NetworkArtist,
    GroupingCollection,
    TreeArtist,
    VertexCollection,
    EdgeCollection,
    LeafEdgeCollection,
    LabelCollection,
    EdgeArrowCollection,
    CascadeCollection,
    Vertex3DCollection,
    Edge3DCollection,
)
