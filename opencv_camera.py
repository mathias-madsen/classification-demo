import numpy as np
import cv2 as cv


class OpenCVCamera(cv.VideoCapture):

    def __init__(self, camera_index, downsampling_factor=None):
        cv.VideoCapture.__init__(self, camera_index)
        self.downsampling_factor = downsampling_factor

    def read_mirrored_rgb(self):
        success, bgr = self.read()
        assert success
        rgb = bgr[:, :, ::-1]
        rgb = rgb[:, ::-1, :]  # mirror left/right for visual sanity
        dsf = self.downsampling_factor
        return rgb[::dsf, ::dsf]  # downsample for speed


if __name__ == "__main__":

    from matplotlib import pyplot as plt

    camera = OpenCVCamera(0)
    image = camera.read_mirrored_rgb()

    plt.ion()
    figure, axes = plt.subplots(figsize=(12, 8))
    panel = axes.imshow(image)
    axes.set_title("0", fontsize=28)
    axes.axis("off")
    figure.tight_layout()

    nread = 0
    while plt.fignum_exists(figure.number):
        nread += 1
        axes.set_title(str(nread), fontsize=28)
        image = camera.read_mirrored_rgb()
        panel.set_data(image)
        plt.pause(0.01)
    
    camera.release()

    