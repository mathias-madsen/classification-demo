import os
import numpy as np
import cv2 as cv
import torch
from matplotlib import pyplot as plt
from time import perf_counter

from encoding.pretrained import OnnxModel


if __name__ == "__main__":

    wrapper = OnnxModel(downsampling_factor=5)
    cam = cv.VideoCapture(0)
    
    result, bgr = cam.read()
    rgb = bgr[:, :, ::-1]
    encoding = wrapper(rgb)

    plt.ion()

    figure, (left, right) = plt.subplots(
        figsize=(12, 6),
        ncols=2,
        width_ratios=(5, 1),
        )

    cam_display = left.imshow(rgb[:, ::-1, :])
    left.axis("off")

    dot_plot, = right.plot([], [], ".")
    right.set_xlim(-10, +10)
    right.set_ylim(-1, len(encoding) + 1)
    right.axis("off")

    figure.tight_layout()

    all_encodings = []
    all_durations = []
    clicks = []
    loop_start = perf_counter()
    while plt.fignum_exists(figure.number):
        start = perf_counter()
        result, bgr = cam.read()
        rgb = bgr[:, :, ::-1]
        encoding = wrapper(rgb)
        all_encodings.append(encoding)
        cam_display.set_data(rgb[:, ::-1, :])
        dot_plot.set_data(encoding, range(len(encoding)))
        plt.pause(1e-4)
        dur = perf_counter() - start
        all_durations.append(1000.0 * dur)
        clicks.append(perf_counter() - loop_start)

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
