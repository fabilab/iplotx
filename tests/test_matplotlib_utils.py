import pytest
import numpy as np
import matplotlib as mpl
import iplotx as ipx


def test_additional_set_methods_noclass():
    attributes = ["xlim", "ylim"]
    res = ipx.utils.matplotlib._additional_set_methods(attributes)
    assert hasattr(res, "__call__")


def test_additional_set_methods_class():
    attributes = ["xlim", "ylim"]

    class DummyArtist:
        pass

    res = ipx.utils.matplotlib._additional_set_methods(attributes, DummyArtist)
    assert hasattr(res, "set_xlim")
    assert hasattr(res, "set_ylim")


def test_midpoint_squared_bezier():
    path = mpl.path.Path(
        vertices=[[0, 0], [1, 0], [1, 1]],
        codes=[mpl.path.Path.MOVETO, mpl.path.Path.CURVE3, mpl.path.Path.CURVE3],
    )
    coord, rot = ipx.utils.matplotlib._compute_mid_coord_and_rot(
        path,
        mpl.transforms.IdentityTransform().transform,
    )
    assert np.allclose(coord, [0.75, 0.25])
    assert np.isclose(rot, np.pi / 4)


def test_midpoint_invalid():
    path = mpl.path.Path(
        vertices=[[0, 0], [1, 0]],
        codes=[mpl.path.Path.MOVETO, mpl.path.Path.MOVETO],
    )
    with pytest.raises(ValueError):
        ipx.utils.matplotlib._compute_mid_coord_and_rot(
            path,
            mpl.transforms.IdentityTransform().transform,
        )


def test_label_size_emptystring():
    width, height = ipx.utils.matplotlib._get_label_width_height("")
    assert width == 0
    assert height == 0


def test_build_cmap_fun():
    res = ipx.utils.matplotlib._build_cmap_fun(
        {"color": 0, "cmap": "Greys"},
        "color",
    )
    assert np.allclose(res(0), np.ones(4))
