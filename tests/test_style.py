from copy import deepcopy
from collections import defaultdict
import pytest

import iplotx as ipx


def test_reset():
    ipx.style.use("hollow")
    ipx.style.reset()
    assert ipx.style.current == ipx.style.styles["default"]


def test_get_nonexistent_style():
    with pytest.raises(TypeError):
        ipx.style.get_style(".vertex.size.")


def test_get_style_invalid():
    with pytest.raises(ValueError):
        ipx.style.get_style("default", {}, "third_arg")


def test_get_style_args():
    assert ipx.style.get_style(".vertex.curved", 80) == 80


def test_vertex_node():
    style = ipx.style.merge_styles(
        [
            dict(vertex_size=80),
            dict(node_label_color="grey"),
        ]
    )
    assert isinstance(style, dict)
    assert "vertex" in style
    assert "node" not in style
    assert "size" in style.get("vertex", {})
    assert "label" in style.get("vertex", {})
    assert "color" in style.get("vertex", {"label": {}}).get("label", {})


def test_edgepadding():
    style = ipx.style.merge_styles(
        [
            {
                "edge": {
                    "padding": 8,
                    "shrink": 2,
                }
            },
        ]
    )
    assert isinstance(style, dict)
    assert "edge" in style
    assert "shrink" in style.get("edge", {})
    assert "padding" not in style.get("edge", {})

    style = ipx.style.merge_styles(
        [
            {
                "edge": {
                    "padding": 8,
                }
            },
        ]
    )
    assert isinstance(style, dict)
    assert "edge" in style
    assert "shrink" in style.get("edge", {})
    assert "padding" not in style.get("edge", {})


def test_rotate_type_fallback():
    style = {
        "size": {"hello": 80},
    }
    assert ipx.style.rotate_style(
        style,
        key="world",
    ) == {"size": 0}


def test_rotate_style_key2():
    style = {
        "size": {"hello": 80},
    }
    assert ipx.style.rotate_style(
        style,
        key="world",
        key2="hello",
    ) == {"size": 80}


def test_rotate_style_noargs():
    style = {
        "size": {"hello": 80},
    }
    with pytest.raises(ValueError):
        ipx.style.rotate_style(style)


def test_use_style_invalid():
    with pytest.raises(TypeError):
        ipx.style.use(["hollow", ".vertex.size."])
    assert ipx.style.current == ipx.style.styles["default"]


def test_flat_style():
    with ipx.style.context(
        dict(
            vertex_size=80,
            edge_label_bbox_facecolor="yellow",
            zorder=80,
        ),
    ):
        current = ipx.style.current
        assert current["vertex"]["size"] == 80
        assert current["edge"]["label"]["bbox"]["facecolor"] == "yellow"

        with ipx.style.context(
            vertex_size=70,
        ):
            assert current["vertex"]["size"] == 70
            assert current["edge"]["label"]["bbox"]["facecolor"] == "yellow"


def test_generator():
    styles = iter([{"vertex_size": 80}, {"vertex_size": 70}])
    with ipx.style.context(
        styles,
    ):
        assert ipx.style.current["vertex"]["size"] == 70


def test_use():
    style = deepcopy(ipx.style.current)
    ipx.style.use("hollow")
    ipx.style.use("default")
    assert style == ipx.style.current


def test_copy_with_deep_values():
    partial_style = defaultdict(lambda: 80, {"a": 10})
    partial_style_copy = ipx.style.copy_with_deep_values(partial_style)
    assert partial_style_copy["a"] == 10
    assert partial_style_copy["b"] == 80
    assert hasattr(partial_style_copy, "default_factory")

    style = {"vertex": {"size": defaultdict(lambda: 80, {"a": 10})}}
    style_copy = ipx.style.copy_with_deep_values(style)
    assert style_copy["vertex"]["size"]["a"] == 10
    assert style_copy["vertex"]["size"]["b"] == 80
    assert not hasattr(style_copy, "default_factory")
    assert hasattr(style_copy["vertex"]["size"], "default_factory")
