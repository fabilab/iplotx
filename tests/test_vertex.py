import unittest
import pytest
import numpy as np
import pandas as pd
from matplotlib import patches
import matplotlib.pyplot as plt
import iplotx as ipx


def vertex_collection(**kwargs):
    if "layout" not in kwargs:
        kwargs["layout"] = pd.DataFrame(
            data=np.array([[0, 0], [1, 1]], np.float64),
            index=["A", "B"],
        )
    return ipx.artists.VertexCollection(**kwargs)


class VertexCollectionTest(unittest.TestCase):
    def setUp(self):
        self.vertex_collection = vertex_collection()

    def test_vertex_count(self):
        self.assertEqual(len(self.vertex_collection), 2)

    def test_vertex_names(self):
        pd.testing.assert_index_equal(self.vertex_collection.get_index(), pd.Index(["A", "B"]))

    def test_layout(self):
        expected_layout = pd.DataFrame(
            data=np.array([[0, 0], [1, 1]], np.float64), index=["A", "B"]
        )
        pd.testing.assert_frame_equal(self.vertex_collection._layout, expected_layout)

    def test_vertex_id(self):
        self.assertEqual(self.vertex_collection.get_vertex_id(0), "A")

    def test_get_style(self):
        with ipx.style.context({"vertex": {"size": 5}}):
            style = ipx.style.get_style(".vertex")
            vc = vertex_collection(style=style)
            assert vc.get_style() == style

    def test_get_sizes(self):
        with ipx.style.context({"vertex": {"size": 5}}):
            vc = vertex_collection(style=ipx.style.get_style(".vertex"))
            sizes = vc.get_sizes()
            np.testing.assert_array_equal(sizes, np.array([5, 5]))
            sizes_dpi = vc.get_sizes_dpi()
            np.testing.assert_array_equal(sizes_dpi, np.array([5, 5]))

    def test_set_sizes(self):
        self.vertex_collection.set_sizes(None)
        np.testing.assert_array_equal(
            self.vertex_collection.get_sizes(), np.array([], dtype=np.float64)
        )

    def test_get_layout(self):
        expected_layout = pd.DataFrame(
            data=np.array([[0, 0], [1, 1]], np.float64), index=["A", "B"]
        )
        pd.testing.assert_frame_equal(self.vertex_collection.get_layout(), expected_layout)

    def test_get_offset(self):
        np.testing.assert_array_equal(
            self.vertex_collection.get_offsets(), np.array([[0, 0], [1, 1]], np.float64)
        )
        np.testing.assert_array_equal(
            self.vertex_collection.get_offsets(ignore_layout=False),
            np.array([[0, 0], [1, 1]], np.float64),
        )

    def test_set_offsets(self):
        self.vertex_collection.set_offsets(np.array([[2, 2], [3, 3]], np.float64))
        np.testing.assert_array_equal(
            self.vertex_collection.get_offsets(),
            np.array([[2, 2], [3, 3]], np.float64),
        )

    def test_update_layout(self):
        vc = vertex_collection()
        vc._offsets = None
        vc._update_offsets_from_layout()
        np.testing.assert_array_equal(
            self.vertex_collection.get_offsets(), np.array([[0, 0], [1, 1]], np.float64)
        )

    def test_updat_layout_unsupported(self):
        vc = vertex_collection()
        vc._layout_coordinate_system = "unsupported"
        with pytest.raises(ValueError):
            vc._update_offsets_from_layout()

    def test_update_layout_polar(self):
        vc = vertex_collection()
        vc._layout_coordinate_system = "polar"
        vc._layout.iloc[:, 0] = [1, 1]
        vc._layout.iloc[:, 1] = [0, np.pi]
        vc._update_offsets_from_layout()
        np.testing.assert_array_almost_equal(
            vc.get_offsets(),
            np.array([[1, 0], [-1, 0]], np.float64),
        )

        vc._offsets = None
        vc._update_offsets_from_layout()
        np.testing.assert_array_almost_equal(
            vc.get_offsets(),
            np.array([[1, 0], [-1, 0]], np.float64),
        )

    @pytest.mark.filterwarnings("error")
    def test_nolabels(self):
        with ipx.style.context({"vertex": {"size": "label"}}):
            style = ipx.style.get_style(".vertex")
            with pytest.raises(UserWarning):
                vertex_collection(style=style)

    @pytest.mark.filterwarnings("ignore")
    def test_nolabels_ignorewarning(self):
        with ipx.style.context({"vertex": {"size": "label"}}):
            style = ipx.style.get_style(".vertex")
            vc = vertex_collection(style=style)
            sizes = vc.get_sizes()
            np.testing.assert_array_almost_equal(sizes, np.array([20] * 2, np.float64), decimal=2)

    def test_no_labels_check(self):
        assert self.vertex_collection._labels is None
        self.assertFalse(hasattr(self.vertex_collection, "_label_collection"))
        assert self.vertex_collection.get_labels() is None

    def test_stale(self):
        self.vertex_collection.stale = True
        self.assertTrue(self.vertex_collection.stale)
        self.vertex_collection.stale = False
        self.assertFalse(self.vertex_collection.stale)

    def test_update_labels(self):
        assert self.vertex_collection._update_labels() is None

    def test_draw_invisible(self):
        self.vertex_collection.set_visible(False)
        fig = plt.figure()
        try:
            assert self.vertex_collection.draw(fig.canvas.get_renderer()) is None
        finally:
            plt.close(fig)

    def test_draw_zerolength(self):
        vc = vertex_collection(layout=pd.DataFrame(data=np.array([[], []]).T, index=[]))
        fig = plt.figure()
        try:
            assert vc.draw(fig.canvas.get_renderer()) is None
        finally:
            plt.close(fig)


class VertexCollectionLabelsTest(unittest.TestCase):
    def setUp(self):
        with ipx.style.context({"vertex": {"size": "label"}}):
            self.vertex_collection = vertex_collection(
                labels=["hello", "world"],
                style=ipx.style.get_style(".vertex"),
            )

    def test_label_creation(self):
        labels = self.vertex_collection.get_labels()
        assert labels is not None
        assert isinstance(labels, ipx.artists.LabelCollection)

    def test_children(self):
        children = self.vertex_collection.get_children()
        assert isinstance(children, tuple)
        assert len(children) == 1
        assert isinstance(children[0], ipx.artists.LabelCollection)

    def test_set_figure(self):
        fig = plt.figure()
        self.vertex_collection.set_figure(fig)
        try:
            assert self.vertex_collection.get_figure() == fig
            assert self.vertex_collection.get_labels().get_figure() == fig
        finally:
            plt.close(fig)

    def test_update_labels(self):
        self.vertex_collection._layout_coordinate_system = "polar"
        assert self.vertex_collection._update_labels() is None


class VertexCollectionCmapTest(unittest.TestCase):
    def setUp(self):
        with ipx.style.context(
            {
                "vertex": {
                    "cmap": plt.cm.viridis,
                    "facecolor": [2, 3],
                },
            }
        ):
            style = ipx.style.get_style(".vertex")
            vc = vertex_collection(style=style)
        self.vertex_collection = vc

    def test_init_cmap(self):
        assert self.vertex_collection.get_cmap() == plt.cm.viridis


def test_shapes():
    for marker in ["o", "c", "s", "^", "v", "<", ">", "d", "p", "h", "8", "e", "*"]:
        patch, size = ipx.vertex.make_patch(marker=marker)
        assert size == 20
        assert isinstance(patch, patches.Patch)

    with pytest.raises(KeyError):
        ipx.vertex.make_patch(marker="invalid_marker")
