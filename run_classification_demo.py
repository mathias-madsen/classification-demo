from argparse import ArgumentParser
from matplotlib import pyplot as plt
from IPython.terminal.embed import InteractiveShellEmbed

from discriminator import BiGaussianDiscriminator
from collector import DataCollector


if __name__ == "__main__":

    parser = ArgumentParser()

    str_or_int = lambda x: int(x) if x.isnumeric() else x
    parser.add_argument("--camera", type=str_or_int, default=0)
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("--downsampling", type=int, default=10)

    args, _ = parser.parse_known_args()

    shell = InteractiveShellEmbed()
    shell.enable_matplotlib()

    plt.ion()

    print("Setting up camera . . .")
    if type(args.camera) == int:
        import cv2 as cv
        camera = cv.VideoCapture(args.camera)
    elif args.camera.lower() == "ximea":
        from ximea_camera import XimeaCamera
        camera = XimeaCamera()
    else:
        raise ValueError("Unrecognized camera option %r" % (args.camera,))

    if args.model == "resnet":
        from image_encoding import ResNet50Encoder
        encoder = ResNet50Encoder(downsampling_factor=args.downsampling)
    elif args.model == "local":
        from pretrained import OnnxModel
        encoder = OnnxModel(downsampling_factor=args.downsampling)
    else:
        raise ValueError("Unrecognized model option %r" % (args.model,))

    print("Encoder: %r\n" % encoder)

    collector = DataCollector(
        image_encoder=encoder,
        discriminator=BiGaussianDiscriminator(),
        camera=camera,
        )

    try:
        shell()
    except:
        collector.on_close()