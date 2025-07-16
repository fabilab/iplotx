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


def make_small_tree(layout="horizontal"):
    tree = next(
        Phylo.NewickIO.parse(
            StringIO(
                "(()(()((()())(()()))))",
            )
        )
    )

    internal_data = ipx.ingest.ingest_tree_data(
        tree,
        layout,
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


def make_cascade(layout="horizontal", orientation="right", style={}):
    small_tree = make_small_tree(layout=layout)
    with ipx.style.context(style):
        return CascadeCollection(
            tree=small_tree["tree"],
            layout=small_tree["layout"],
            layout_name=layout,
            orientation=orientation,
            style=ipx.style.get_style(".cascade"),
            provider=small_tree["provider"],
            transform=mpl.transforms.IdentityTransform(),
        )


def test_init():
    cascade = make_cascade()
    assert isinstance(cascade, CascadeCollection)


def test_init_unsupported():
    with pytest.raises(ValueError):
        make_cascade(layout="unsupported")


def test_update_maxdepth():
    for layout, orientation in [
        ("horizontal", "right"),
        ("horizontal", "left"),
        ("vertical", "ascending"),
        ("vertical", "descending"),
        ("radial", "clockwise"),
        ("radial", "counterclockwise"),
    ]:
        cascade = make_cascade(layout=layout, orientation=orientation)
        cascade._update_maxdepth()


def test_update_maxdepth_unsupported():
    cascade = make_cascade(layout="horizontal", orientation="unsupported")
    with pytest.raises(ValueError):
        cascade._update_maxdepth()


def test_extend_true():
    make_cascade(
        style={
            "cascade": {
                "extend": True,
            }
        }
    )


def test_extend_leaflabels():
    make_cascade(
        style={
            "cascade": {
                "extend": "leaf_labels",
            },
        },
    )
