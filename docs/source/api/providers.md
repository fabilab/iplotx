# Data provider protocols
```{note}
  The `NetworkDataProvider` protocol requires individual providers to implement `__call__` which actually processed the network, layout, and label data. We are working on simplifying this situation.

  The `TreeDataProvider` protocol is simpler to implement as this function is standardised and needs not be implemented by individual providers. Only the support functions, which are very simple and small, needs to be implemented by tree providers.
```


```{eval-rst}
.. autoclass:: iplotx.ingest.typing.NetworkDataProvider
    :members:

.. autoclass:: iplotx.ingest.typing.TreeDataProvider
    :members:

```
