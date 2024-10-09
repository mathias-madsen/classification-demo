from matplotlib import pyplot as plt
from IPython.terminal.embed import InteractiveShellEmbed

from discriminator import BiGaussianDiscriminator
from collector import DataCollector


if __name__ == "__main__":

    shell = InteractiveShellEmbed()
    shell.enable_matplotlib()

    plt.ion()

    image_source = 0
    encoder = "resnet"

    print("Setting up camera . . .")
    if type(image_source) == int:
        import cv2 as cv
        camera = cv.VideoCapture(image_source)
    else:
        from ximea_camera import XimeaCamera
        camera = XimeaCamera()

    if encoder == "resnet":
        from image_encoding import ResNet50Encoder
        encode = ResNet50Encoder()
    elif encoder == "local":
        from pretrained import encode

    discriminator = BiGaussianDiscriminator()
    collector = DataCollector(
        image_encoder=encode,
        discriminator=discriminator,
        camera=camera,
        )

    try:
        shell()
    except:
        collector.on_close()