import pytest
import iplotx as ipx


def test_incomplete_network_protocol():
    class IncompleteProtocol(ipx.ingest.typing.NetworkDataProvider):
        pass

    protocol = IncompleteProtocol(None)

    with pytest.raises(NotImplementedError):
        protocol()

    with pytest.raises(NotImplementedError):
        protocol.check_dependencies()

    with pytest.raises(NotImplementedError):
        protocol.graph_type()

    with pytest.raises(NotImplementedError):
        protocol.is_directed()

    with pytest.raises(NotImplementedError):
        protocol.number_of_vertices()


def test_incomplete_tree_protocol():
    class IncompleteProtocol(ipx.ingest.typing.TreeDataProvider):
        pass

    protocol = IncompleteProtocol(None)

    with pytest.raises(NotImplementedError):
        protocol.check_dependencies()

    with pytest.raises(NotImplementedError):
        protocol.tree_type()

    with pytest.raises(NotImplementedError):
        protocol.get_leaves()

    with pytest.raises(NotImplementedError):
        protocol.preorder()

    with pytest.raises(NotImplementedError):
        protocol.postorder()

    with pytest.raises(NotImplementedError):
        protocol.get_children(None)

    with pytest.raises(NotImplementedError):
        protocol.get_branch_length(None)
