import os
import numpy as np
import cv2 as cv
import torch
from PIL import Image
from matplotlib import pyplot as plt

from image_encoding import preprocess, model


folder = os.path.dirname(os.path.abspath(__file__))
names_file = os.path.join(folder, "imagenet_classes.txt")
with open(names_file, "r") as f:
    IMAGENET_CLASS_NAMES = [s.strip() for s in f.readlines()]


if __name__ == "__main__":

    cam = cv.VideoCapture(0)
    
    result, bgr = cam.read()
    rgb = bgr[:, :, ::-1]

    input_image = Image.fromarray(rgb[:, ::-1, :])
    input_tensor = preprocess(input_image)
    with torch.no_grad():
        logits, encoding = model(input_tensor.unsqueeze(0))
        logits = logits.squeeze(0)
        encoding = encoding.squeeze(0)

    plt.ion()

    figure, (left, right) = plt.subplots(figsize=(12, 6),
                                         ncols=2,
                                         width_ratios=(5, 1))

    cam_display = left.imshow(bgr[:, ::-1, ::-1])
    left.axis("off")

    dot_plot, = right.plot([], [], ".")
    figure.suptitle("|||||", fontsize=48)
    right.set_xlim(-10, +10)
    right.set_ylim(-1, len(encoding) + 1)
    right.axis("off")

    figure.tight_layout()

    while plt.fignum_exists(figure.number):

        result, bgr = cam.read()
        rgb = bgr[:, :, ::-1]

        input_image = Image.fromarray(rgb[:, ::-1, :])
        input_tensor = preprocess(input_image)
        with torch.no_grad():
            logits, encoding = model(input_tensor.unsqueeze(0))
            logits = logits.squeeze(0)
            encoding = encoding.squeeze(0)
        decreasing = torch.argsort(logits, descending=True)
        probs = torch.softmax(logits, -1)
        winners = [IMAGENET_CLASS_NAMES[i] for i in decreasing[:3]]
        winning_probs = [float(probs[i]) for i in decreasing[:3]]
        print(dict(zip(winners, winning_probs)))

        cam_display.set_data(bgr[:, ::-1, ::-1])
        dot_plot.set_data(encoding, range(len(encoding)))

        title = ", ".join("%r: %.1f pct" % (name, 100 * prob)
                          for name, prob in zip(winners, winning_probs))
        figure.suptitle(title, fontsize=24)

        plt.pause(1 / 12)
