import os
import numpy as np
import cv2 as cv
import torch
from PIL import Image
from matplotlib import pyplot as plt
from encoding.image_encoding import ResNet50Encoder


folder = os.path.dirname(os.path.abspath(__file__))
names_file = os.path.join(folder, "imagenet_classes.txt")
with open(names_file, "r") as f:
    IMAGENET_CLASS_NAMES = [s.strip() for s in f.readlines()]


if __name__ == "__main__":

    wrapper = ResNet50Encoder()
    cam = cv.VideoCapture(0)
    
    result, bgr = cam.read()
    rgb = bgr[:, :, ::-1]
    logits, encoding = wrapper(rgb, return_logits=True)

    plt.ion()

    figure, (left, right) = plt.subplots(figsize=(12, 6),
                                         ncols=2,
                                         width_ratios=(5, 1))

    cam_display = left.imshow(rgb[:, ::-1, :])
    left.axis("off")

    dot_plot, = right.plot([], [], ".")
    figure.suptitle("|||||", fontsize=48)
    right.set_xlim(-10, +10)
    right.set_ylim(-1, len(encoding) + 1)
    right.axis("off")

    figure.tight_layout()

    all_encodings = []

    while plt.fignum_exists(figure.number):

        result, bgr = cam.read()
        rgb = bgr[:, :, ::-1]
        logits, encoding = wrapper(rgb, return_logits=True)
        all_encodings.append(encoding)
        increasing = np.argsort(logits)
        decreasing = increasing[::-1]
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        winners = [IMAGENET_CLASS_NAMES[i] for i in decreasing[:3]]
        winning_probs = [float(probs[i]) for i in decreasing[:3]]
        print({k: "%.5f" % v for k, v in zip(winners, winning_probs)})

        cam_display.set_data(rgb[:, ::-1, :])
        dot_plot.set_data(encoding, range(len(encoding)))

        title = ", ".join("%r: %.1f pct" % (name, 100 * prob)
                          for name, prob in zip(winners, winning_probs))
        figure.suptitle(title, fontsize=24)

        plt.pause(1 / 12)

    means = np.mean(all_encodings, axis=0)
    stds = np.std(all_encodings, axis=0)

    print("Encoding dimensionality: %s" % len(means))
    print("encoding means 90 pct range: %s" % np.percentile(means, [5, 95]))
    print("encoding stds 90 pct range: %s" % np.percentile(stds, [5, 95]))
    print()