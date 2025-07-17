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
