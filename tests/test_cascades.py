from io import StringIO
import pytest
import unittest
import importlib
import matplotlib as mpl
import iplotx as ipx

if importlib.util.find_spec("Bio") is None:
    raise unittest.SkipTest("biopython not found, skipping tests")
else:
    from Bio import Phylo


CascadeCollection = ipx.artists.CascadeCollection


@pytest.fixture
def small_tree():
    tree = next(
        Phylo.NewickIO.parse(
            StringIO(
                "(()(()((()())(()()))))",
            )
        )
    )

    internal_data = ipx.ingest.ingest_tree_data(
        tree,
        "horizontal",
        directed=False,
        layout_style={},
    )
    layout = internal_data["vertex_df"]
    provider = ipx.ingest.data_providers["tree"][internal_data["tree_library"]]

    return {
        "tree": tree,
        "layout": layout,
        "provider": provider,
    }


def test_init(small_tree):
    CascadeCollection(
        tree=small_tree["tree"],
        layout=small_tree["layout"],
        layout_name="horizontal",
        orientation="right",
        style=ipx.style.get_style(".cascade"),
        provider=small_tree["provider"],
        transform=mpl.transforms.IdentityTransform(),
    )


def test_update_maxdepth(small_tree):
    cascades = CascadeCollection(
        tree=small_tree["tree"],
        layout=small_tree["layout"],
        layout_name="horizontal",
        orientation="right",
        style=ipx.style.get_style(".cascade"),
        provider=small_tree["provider"],
        transform=mpl.transforms.IdentityTransform(),
    )
    cascades._update_maxdepth()
