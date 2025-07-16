import pytest
import numpy as np
import matplotlib as mpl
from iplotx.edge import ports


def test_port_angles():
    trans_inv = mpl.transforms.IdentityTransform().transform
    for portstring, expected in ports.port_dict.items():
        res = ports._get_port_unit_vector(portstring, trans_inv)
        np.testing.assert_array_almost_equal(res, expected, decimal=6)


def test_port_angles_inverted():
    trans_inv = mpl.transforms.Affine2D().scale(-1, -1).transform
    port_dict = {
        "e": ports.port_dict["w"],
        "n": ports.port_dict["s"],
        "w": ports.port_dict["e"],
        "s": ports.port_dict["n"],
        "nw": ports.port_dict["se"],
        "sw": ports.port_dict["ne"],
        "ne": ports.port_dict["sw"],
        "se": ports.port_dict["nw"],
    }
    for portstring, expected in port_dict.items():
        res = ports._get_port_unit_vector(portstring, trans_inv)
        np.testing.assert_array_almost_equal(res, expected, decimal=6)


def test_port_unsupported():
    trans_inv = mpl.transforms.IdentityTransform().transform
    with pytest.raises(KeyError):
        ports._get_port_unit_vector("unsupported", trans_inv)
