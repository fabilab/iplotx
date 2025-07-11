"""
All artists defined in iplotx.
"""

from .network import NetworkArtist
from .tree import TreeArtist
from .vertex import VertexCollection
from .edge import EdgeCollection
from .label import LabelCollection
from .edge.arrow import EdgeArrowCollection
from .edge.leaf import LeafEdgeCollection
from .cascades import CascadeCollection


___all__ = (
    NetworkArtist,
    TreeArtist,
    VertexCollection,
    EdgeCollection,
    LeafEdgeCollection,
    LabelCollection,
    EdgeArrowCollection,
    CascadeCollection,
)
