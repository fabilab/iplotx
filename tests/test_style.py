from copy import deepcopy
from collections import defaultdict
import unittest

import iplotx as ipx


class StyleTestRunner(unittest.TestCase):
    def test_flat_style(self):
        with ipx.style.context(
            dict(
                vertex_size=80,
                edge_label_bbox_facecolor="yellow",
            ),
        ):
            current = ipx.style.current
            self.assertEqual(current["vertex"]["size"], 80)
            self.assertEqual(current["edge"]["label"]["bbox"]["facecolor"], "yellow")

            with ipx.style.context(
                vertex_size=70,
            ):
                self.assertEqual(current["vertex"]["size"], 70)
                self.assertEqual(current["edge"]["label"]["bbox"]["facecolor"], "yellow")

    def test_generator(self):
        styles = iter([{"vertex_size": 80}, {"vertex_size": 70}])
        with ipx.style.context(
            styles,
        ):
            self.assertEqual(ipx.style.current["vertex"]["size"], 70)

    def test_use(self):
        style = deepcopy(ipx.style.current)
        ipx.style.use("hollow")
        ipx.style.use("default")
        self.assertEqual(style, ipx.style.current)

    def test_copy_with_deep_values(self):
        partial_style = defaultdict(lambda: 80, {"a": 10})
        partial_style_copy = ipx.style.copy_with_deep_values(partial_style)
        self.assertEqual(partial_style_copy["a"], 10)
        self.assertEqual(partial_style_copy["b"], 80)
        self.assertTrue(hasattr(partial_style_copy, "default_factory"))

        style = {"vertex": {"size": defaultdict(lambda: 80, {"a": 10})}}
        style_copy = ipx.style.copy_with_deep_values(style)
        self.assertEqual(style_copy["vertex"]["size"]["a"], 10)
        self.assertEqual(style_copy["vertex"]["size"]["b"], 80)
        self.assertFalse(hasattr(style_copy, "default_factory"))
        self.assertTrue(hasattr(style_copy["vertex"]["size"], "default_factory"))


def suite():
    return unittest.TestSuite(
        [
            unittest.defaultTestLoader.loadTestsFromTestCase(StyleTestRunner),
        ]
    )


def test():
    runner = unittest.TextTestRunner()
    runner.run(suite())


if __name__ == "__main__":
    test()
