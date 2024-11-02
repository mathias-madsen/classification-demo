import os
from argparse import ArgumentParser
from tempfile import TemporaryDirectory
import cv2 as cv
from tqdm import tqdm
import json

from discriminator import BiGaussianDiscriminator
from dataset import EncodingData
from gaussians.moments_tracker import combine


def iter_video_relpaths(folder, extension=".avi"):
    for root, _, filenames in os.walk(folder):
        for fn in filenames:
            if fn.endswith(extension):
                abspath = os.path.join(root, fn)
                yield os.path.relpath(abspath, folder)


def compile_video_paths_and_index_dict(folder, extension=".avi"):
    relpaths = list(iter_video_relpaths(folder, extension))
    # sort by file name, and thus by age:
    relpaths = sorted(relpaths, key=lambda rp: os.path.basename(rp))
    abspaths = [os.path.join(folder, rp) for rp in relpaths]
    indices = [int(os.path.dirname(rp)) for rp in relpaths]
    return dict(zip(abspaths, indices))


def iter_video(video_path, verbose=True):
    video = cv.VideoCapture(video_path)
    length = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    if verbose:
        frame_indices = tqdm(range(length))
    else:
        frame_indices = range(length)
    for _ in frame_indices:
        success, bgr = video.read()
        assert success
        yield bgr[:, :, ::-1]


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "indir",
        help="The folder containing the class videos",
        type=str,
        )

    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--downsampling", type=int, default=-1)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--class_name_0", type=str, default="")
    parser.add_argument("--class_name_1", type=str, default="")

    args, _ = parser.parse_known_args()

    print("Received arguments\n")
    for key, value in args.__dict__.items():
        print("    %r: %r" % (key, value))
    print()

    # if model info is stored and you need it, load it from file:
    model_info_path = os.path.join(args.indir, "model_info.json")
    if os.path.isfile(model_info_path):
        with open(model_info_path, "rt") as source:
            model_args = json.load(source)
        if not args.model:
            args.model = model_args.pop("class")
    else:
        assert args.model
        model_args = {}

    # create image encoder:
    if args.model == "resnet":
        from image_encoding import ResNet50Encoder
        downsampling_factor = model_args.get("downsampling_factor", 5)
        encoder = ResNet50Encoder(downsampling_factor=downsampling_factor)
    elif args.model == "local" or args.model == "OnnxModel":
        from pretrained import OnnxModel
        downsampling_factor = model_args.get("downsampling_factor", 5)
        keys = model_args.get("keys", ["Eyx1", "Varsyx1", "Covyx1"])
        encoder = OnnxModel(downsampling_factor=downsampling_factor, keys=keys)
    else:
        raise ValueError("Unrecognized model option %r" % (args.model,))
    
    # tell the user what you did (the `repr`` will also print the kwargs):
    print("Encoder: %r\n" % encoder)

    # create a temporary outdir if one isn't given:
    if not args.outdir:
        tempdir = TemporaryDirectory()
        args.outdir = tempdir.name

    # initialize empty data set:
    dataset = EncodingData(encoder, rootdir=args.outdir)

    # get or potentially overwrite class names:
    if not args.class_name_0 or not args.class_name_1:
        try:
            dataset.load_class_names_from_file(args.indir)
        except FileNotFoundError:
            raise ValueError(
                "Found not file of stored class names and did not receive "
                "command-line arguments '--class_name_0' or '--class_name_1'"
                )
    else:
        dataset.class_names[0] = args.model_class_0
        dataset.class_names[1] = args.model_class_1

    # compile a dict of video paths with class indices:
    pathdict = compile_video_paths_and_index_dict(args.indir)

    # and tell the user what you found:
    print("Found %s video files:" % len(pathdict))
    for i in sorted(set(pathdict.values())):
        count = sum(j == i for j in pathdict.values())
        print("%r: %s" % (dataset.class_names[i], count))
    print()
    

    # get any image to sniff out the latent dim:
    first_vidpath = next(iter(pathdict.keys()))
    first_image = next(iter_video(first_vidpath, verbose=False))
    dataset.compute_dimensions(first_image)

    # load and encode the video files one by one:
    for vidpath, index in pathdict.items():
        # class_name = dataset.class_names[index]
        # print("Loading %r (class %r). . ." % (vidpath, class_name))
        dataset.initialize_recording(index)
        for frame in iter_video(vidpath):
            dataset.record_frame(frame)
        dataset.save_recording()
    print()

    # once you've collected all the codes, cross-validate:

    discriminator = BiGaussianDiscriminator()

    print("CROSS-VALIDING CLASS %r\n" % dataset.class_names[0])
    k0 = 0
    n0 = 0
    for eps_idx, test_codes in enumerate(dataset.class_episode_codes[0]):
        stats_0 = dataset.class_episode_stats[0].copy()
        stats_0.pop(eps_idx)
        stats_1 = dataset.class_episode_stats[1]
        discriminator.fit_with_moments(combine(stats_1), combine(stats_0))
        corrects = discriminator(test_codes) < 0.0
        k = sum(corrects)
        n = len(corrects)
        print("Episode %r: %s/%s = %.1f pct" % (eps_idx, k, n, 100 * k / n))
        k0 += k
        n0 += n
    print("Class average: %s/%s = %.1f pct" % (k0, n0, 100 * k0 / n0))
    print()

    print("CROSS-VALIDING CLASS %r\n" % dataset.class_names[1])
    k1 = 0
    n1 = 0
    for eps_idx, test_codes in enumerate(dataset.class_episode_codes[1]):
        stats_0 = dataset.class_episode_stats[0]
        stats_1 = dataset.class_episode_stats[1].copy()
        stats_1.pop(eps_idx)
        discriminator.fit_with_moments(combine(stats_1), combine(stats_0))
        corrects = discriminator(test_codes) > 0.0
        k = sum(corrects)
        n = len(corrects)
        print("Episode %r: %s/%s = %.1f pct" % (eps_idx, k, n, 100 * k / n))
        k1 += k
        n1 += n
    print("Class average: %s/%s = %.1f pct" % (k1, n1, 100 * k1 / n1))
    print()

    k = k0 + k1
    n = n0 + n1
    print("Grand average: %s/%s = %.1f pct" % (k, n, 100 * k / n))
    print()
