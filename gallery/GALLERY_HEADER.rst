Gallery
=======
.. warning::
  If plots do not show on screen, try adding `show=True` to your function call:

  .. code-block::

    ipx.network(
      ...,
      show=True,
    )

  This is a peculiarity of `matplotlib`_, the plotting engine behind `iplotx`,
  and depends on your Python environment (e.g., Jupyter notebooks, terminal,
  interactive mode, backend, etc). You might want to enable `interactive mode`_
  *before* plotting to ensure everything is rendered immediately:

  .. code-block::

    import matplotlib.pyplot as plt
    import iplotx as ipx

    ...

    plt.ion()
    ipx.network(
      ...,
    )

  If interactive mode is on, you do not need `show=True` (because plots are
  always rendered immediately anyway).

.. _matplotlib: https://matplotlib.org/
.. _interactive mode: https://matplotlib.org/stable/users/explain/figure/interactive.html
