import os
from copy import deepcopy
import unittest
import pytest
import numpy as np

import iplotx as ipx


class StyleTestRunner(unittest.TestCase):
    def test_flat_style(self):
        style = deepcopy(ipx.style.current)

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
                self.assertEqual(
                    current["edge"]["label"]["bbox"]["facecolor"], "yellow"
                )

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
