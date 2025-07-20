"""
Make banner image for the package.
"""

import numpy as np
import pathlib
from matplotlib import patches
import matplotlib.pyplot as plt


def squareify(image):
    height, width, ncols = image.shape
    dmax = max(height, width)
    new = np.zeros((dmax, dmax, ncols), dtype=image.dtype)
    if height > width:
        diff = (height - width) // 2
        new[:, diff : diff + width, :] = image
        new[:, : diff + 3, :] = 1
        new[:, diff + width - 3 :] = 1
    elif width > height:
        diff = (width - height) // 2
        new[diff : diff + height, :, :] = image
        new[: diff + 3, :, :] = 1
        new[diff + height - 3 :, :, :] = 1

    zoom = 250
    diff = (new.shape[0] - zoom) // 2
    new = new[diff : diff + zoom, diff : diff + zoom, :]

    return new


if __name__ == "__main__":
    # Select random ones to fill a grid
    grid_shape = (4, 6)
    nimages = grid_shape[0] * grid_shape[1]

    # Folder with the images to patchwork
    image_folder_path = pathlib.Path(__file__).parent.parent / "docs" / "build" / "_images"
    images_all = list(image_folder_path.glob("*_thumb.png"))
    images = [images_all[i] for i in np.random.choice(len(images_all), nimages, replace=False)]

    k = 2.0
    kx, ky = 1.13, 0.98
    fig, ax = plt.subplots(figsize=(k * grid_shape[1], k * grid_shape[0]))
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            ij = grid_shape[1] * i + j
            print(ij)
            img = plt.imread(images[ij])
            xmin = kx * (j + 0.5 * int(i % 2))
            xmax = xmin + kx * 1
            ymin = ky * (grid_shape[0] - 1 - i)
            ymax = ymin + ky * 1
            center = xmin + 0.5, ymin + 0.5
            patch = patches.RegularPolygon(
                center,
                6,
                radius=0.65,
                transform=ax.transData,
                facecolor="none",
                edgecolor="black",
                linewidth=1.5,
            )
            ax.add_patch(patch)
            # ax.text(
            #    center[0], center[1], str(ij), ha="center", va="center", fontsize=8, color="black"
            # )
            # Extend to square
            img = squareify(img)
            im = ax.imshow(img, extent=(xmin, xmax, ymin, ymax), aspect=1.0)
            im.set_clip_path(patch)
    ax.set(
        xlim=(-0.2 * kx, kx * (grid_shape[1] + 0.5 * int(grid_shape[0] > 1))),
        ylim=(-0.2 * ky, ky * (grid_shape[0] + 0.2)),
        aspect=1.0,
    )
    ax.set_axis_off()
    plt.ion()
    plt.show()

    fig.savefig(image_folder_path.parent.parent / "source" / "_static" / "banner.png", dpi=100)
