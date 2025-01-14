from matplotlib import pyplot as plt
from argparse import ArgumentParser

from encoding.pretrained import OnnxModel
from encoding.image_encoding import ResNet50Encoder

from classification.discriminator import BiGaussianDiscriminator
from gui.collector import ProbabilityBars


if __name__ == "__main__":

    parser = ArgumentParser()

    str_or_int = lambda x: int(x) if x.isnumeric() else x

    parser.add_argument(
        "discriminator",
        type=str,
        help=(
            "Path to an NPZ file with saved discriminator parameters."
            ),
        )

    parser.add_argument(
        "--camera",
        type=str_or_int,
        default=0,
        help=(
            "The camera to use as a source for the data collection; "
            "use 0, 1, 2, ... to select a built-in camera and 'ximea' "
            "to select a Ximea camera."
            ),
        )

    parser.add_argument(
        "--model",
        type=str,
        choices=["resnet", "local"],
        default="resnet",
        help=(
            "A choice of neural network to use for image encoding, "
            "either a local model saved to a file named `model.onnx` "
            "or a pretrained ResNet50 model."
            ),
        )

    parser.add_argument(
        "--downsampling",
        type=int,
        default=5,
        help=(
            "A downsampling factor for the vectors of image encodings "
            "(not for the image, which is automatically downsampled)."
        ),
        )

    parser.add_argument(
        "--keys",
        type=str,
        nargs="*",
        default=["Eyx1", "Varsyx1", "Covyx1"],
        help=(
            "A choice of outputs from the local model to include in "
            "image encoding. The selected outputs are flattened and "
            "concatenated into a vector which represents the image. "
            "If no value is provided, we try to fall back on the "
            "choice in the 'model_info.json' file, and otherwise "
            "use the default of `['Eyx1', 'Varsyx1', 'Covyx1']`"
            ),
        )

    args, _ = parser.parse_known_args()

    discriminator_path = "/Users/mathias/Desktop/commadrop/discriminator.npz"

    plt.ion()

    print("Setting up camera . . .")
    if type(args.camera) == int:
        from gui.opencv_camera import OpenCVCamera
        camera = OpenCVCamera(args.camera, downsampling_factor=4)
    elif args.camera.lower() == "ximea":
        from gui.ximea_camera import XimeaCamera
        camera = XimeaCamera(downsampling_factor=4)
    else:
        raise ValueError("Unrecognized camera option %r" % (args.camera,))

    if args.model == "resnet":
        from encoding.image_encoding import ResNet50Encoder
        encoder = ResNet50Encoder(downsampling_factor=args.downsampling)
    elif args.model == "local":
        from encoding.pretrained import OnnxModel
        encoder = OnnxModel(
            downsampling_factor=args.downsampling,
            keys=args.keys,
            )
    else:
        raise ValueError("Unrecognized model option %r" % (args.model,))

    print("Encoder: %r\n" % encoder)

    discriminator = BiGaussianDiscriminator.fromsaved(args.discriminator)

    image = camera.read_mirrored_rgb()

    plt.ion()

    figure, (top, bot) = plt.subplots(
        nrows=2,
        figsize=(12, 8),
        height_ratios=[9, 1],
        )

    bars = ProbabilityBars(bot, names=["CLASS 1", "CLASS 2"])
    tv = top.imshow(image)
    top.axis("off")
    plt.tight_layout()
    for _ in range(120):
        image = camera.read_mirrored_rgb()
        tv.set_data(image)
        code = encoder(image)
        logprobs = discriminator(code)
        bars.set_value(logprobs)
        plt.pause(0.001)
        if not plt.fignum_exists(figure.number):
            break
    
    camera.close()