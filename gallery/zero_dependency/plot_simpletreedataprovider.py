"""
Zero-dependency trees
=====================

This example demonstrates how to visualise trees using `iplotx`'s internal `SimpleTree`,
a simple tree representation via the ``children`` attribute.

.. tip::
    This provider is not as powerful as proper tree libraries such as ``cogent3`` or ``ETE4``. It
    can be especially useful when you do not want to add dependencies or for educational purposes.
"""

import iplotx as ipx

tree = {
    "children": (
        {},
        {
            "children": (
                {
                    "children": (
                        {"branch_length": 1.5},
                        {}
                    )
                },
                {
                    "children": (
                        {},
                        {}
                    )
                }
            )
        }
    )
}

# Convert to our simple tree data structure
tree = ipx.ingest.providers.tree.simple.SimpleTree.from_dict(tree)

ipx.tree(
    tree,
)

# %%
# .. note::
#     ``iplotx`` does not generally provide network or tree data structures.
#     This is an exception to demonstrate how to visualise trees without
#     external dependencies.
