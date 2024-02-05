import cv2 as cv
from matplotlib import pyplot as plt


if __name__ == "__main__":

    cam = cv.VideoCapture(0)

    WAITING_TEXT = (
        "NO IMAGE YET\n"
        "YOU MAY NEED TO GRANT THIS SCRIPT\n"
        "PERMISSION TO ACCESS THE CAMERA"
        )

    print("Opening figure . . .")
    plt.ion()
    figure, axes = plt.subplots(figsize=(12, 7))
    
    print("Adding placeholder text . . .")
    axes.text(0.5,
            0.5,
            WAITING_TEXT,
            color="red",
            ha="center",
            va="center",
            fontsize=24,
            fontweight="bold")
    axes.axis("off")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    figure.tight_layout()

    old_result = False
    while plt.fignum_exists(figure.number):
        new_result, bgr = cam.read()
        if new_result:
            if not old_result:
                print("Got first positive result:")
                print("image shape %r" % (bgr[:, ::-1, ::-1].shape,))
                axes.cla()  # remove the text label
                imshow = axes.imshow(bgr[:, ::-1, ::-1])
                axes.axis("off")
                figure.tight_layout()
            else:
                print("image shape %r" % (bgr[:, ::-1, ::-1].shape,))
                imshow.set_data(bgr[:, ::-1, ::-1])
        plt.pause(1 / 24)
        old_result = new_result