import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import patch_collection_2d_to_3d
import matplotlib.pyplot as plt


class Network3DArtist(mpl.artist.Artist):
    """A mock 3D artist for testing purposes."""

    def __init__(self):
        super().__init__()

        xs = np.array([0, 0, 0, 0, 0, 1])
        ys = np.array([0, 0, 0, 0, 1, 1])
        zs = np.array([0, 0.2, 0.4, 1, 1, 1])

        self.add_vertices(xs, ys, zs)

    def add_vertices(self, xs, ys, zs, **kwargs):
        paths = []
        for x, y in zip(xs, ys):
            path = mpl.patches.Circle((0, 0), radius=1).get_path()
            paths.append(path)
            break

        col = mpl.collections.PathCollection(
            paths,
            sizes=10 * np.ones(len(paths)),
            offsets=np.column_stack((xs, ys)),
            transform=mpl.transforms.IdentityTransform(),
        )
        patch_collection_2d_to_3d(col, zs=zs, zdir="z")
        self._vertices = col

    def get_children(self):
        children = []
        if hasattr(self, "_vertices"):
            children.append(self._vertices)

        return tuple(children)

    def draw(self, renderer):
        print("Drawing Network3DArtist")
        for child in self.get_children():
            if child.axes is not None:
                child.do_3d_projection()
            child.draw(renderer)

    def set_figure(self, fig):
        self.figure = fig
        for child in self.get_children():
            child.set_figure(fig)

    @property
    def axes(self):
        return mpl.artist.Artist.axes.__get__(self)

    @axes.setter
    def axes(self, new_axes):
        mpl.artist.Artist.axes.__set__(self, new_axes)
        for child in self.get_children():
            child.axes = new_axes


if __name__ == "__main__":
    fig = plt.figure()

    ax = fig.add_subplot(121, projection="3d")
    na = Network3DArtist()
    print("Before offset transform")
    na._vertices.set_offset_transform(ax.transData)
    print("Before set figure")
    na.set_figure(fig)
    print("Before set axes")
    na.axes = ax
    print("Before add artist")
    ax.add_artist(na)
    ax.set(xlim=(-1, 2), ylim=(-1, 2), zlim=(-1, 2))

    ax2 = fig.add_subplot(122, projection="3d")
    na2 = Network3DArtist()
    na2._vertices.set_offset_transform(ax2.transData)
    ax2.add_collection(na2._vertices)
    ax2.set(xlim=(-1, 2), ylim=(-1, 2), zlim=(-1, 2))

    plt.ion()
    plt.show()
