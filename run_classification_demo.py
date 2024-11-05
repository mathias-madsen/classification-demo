from argparse import ArgumentParser
from matplotlib import pyplot as plt
from IPython.terminal.embed import InteractiveShellEmbed
from tempfile import TemporaryDirectory

from classification.discriminator import BiGaussianDiscriminator
from gui.collector import DataCollector


if __name__ == "__main__":

    parser = ArgumentParser()

    str_or_int = lambda x: int(x) if x.isnumeric() else x
    parser.add_argument("--camera", type=str_or_int, default=0)
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("--downsampling", type=int, default=5)

    args, _ = parser.parse_known_args()

    shell = InteractiveShellEmbed()
    shell.enable_matplotlib()

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
        encoder = OnnxModel(downsampling_factor=args.downsampling)
    else:
        raise ValueError("Unrecognized model option %r" % (args.model,))

    print("Encoder: %r\n" % encoder)

    with TemporaryDirectory() as tempdir:
        collector = DataCollector(
            image_encoder=encoder,
            discriminator=BiGaussianDiscriminator(),
            camera=camera,
            rootdir=tempdir,
            )
        try:
            shell()
        except KeyboardInterrupt:
            collector.on_close()