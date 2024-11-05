import os
import numpy as np
import cv2 as cv
import torch
from PIL import Image
from matplotlib import pyplot as plt
from encoding.pretrained import OnnxModel


if __name__ == "__main__":

    wrapper = OnnxModel(downsampling_factor=None)
    cam = cv.VideoCapture(0)
    
    result, bgr = cam.read()
    rgb = bgr[:, :, ::-1]
    encoding = wrapper(rgb)

    plt.ion()

    figure, (left, right) = plt.subplots(figsize=(12, 6),
                                         ncols=2,
                                         width_ratios=(5, 1))

    cam_display = left.imshow(rgb[:, ::-1, :])
    left.axis("off")

    dot_plot, = right.plot([], [], ".")
    right.set_xlim(-10, +10)
    right.set_ylim(-1, len(encoding) + 1)
    right.axis("off")

    figure.tight_layout()

    all_encodings = []

    while plt.fignum_exists(figure.number):

        result, bgr = cam.read()
        rgb = bgr[:, :, ::-1]
        encoding = wrapper(rgb)
        all_encodings.append(encoding)

        cam_display.set_data(rgb[:, ::-1, :])
        dot_plot.set_data(encoding, range(len(encoding)))

        plt.pause(1 / 12)

    means = np.mean(all_encodings, axis=0)
    vars = np.var(all_encodings, axis=0)

    print("Encoding dimensionality: %s" % len(means))
    print("encoding means 80 pct range: %s" %
          np.percentile(means, [10, 90]))
    print("encoding stds 80 pct range: %s" %
          np.percentile(vars ** 0.5, [10, 90]))
    print()

    print("grand mean:", means.mean())
    print("grand var:", vars.mean())
    print("grand std:", vars.mean() ** 0.5)
    print()
