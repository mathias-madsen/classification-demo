import os
import sys
from argparse import ArgumentParser
from tempfile import TemporaryDirectory
import cv2 as cv
from tqdm import tqdm
import json
from inspect import signature
from time import perf_counter

from classification.dataset import EncodingData


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
        type=str,
        help=("A folder containing a collection of videos organized into "
              "two subfolders named '0' and '1'. The folder may optionally "
              "also contain a 'class_indices_and_names.csv' file and a "
              "'model_info.json' file with fallback choices.")
        )

    parser.add_argument(
        "--model",
        type=str,
        choices=["resnet", "local", "OnnxModel"],
        default="",
        help=("An optional choice of model class, either the pretrained "
              "ResNet or the pretrained ONNX file on disk. If no value is "
              "passed, we will try to fall back on the choice made in the "
              "'model_info.json' file. 'OnnxModel' and 'local' mean the "
              "same thing.")
        )

    parser.add_argument(
        "--downsampling_factor",
        type=int,
        default=-1,
        help=("A downsampling factor **for the vector of encodings** "
              "(not for the input image). If no value is passed, we "
              "fall back on the choice in the 'model_info.json' file, "
              "and if that fails, fall back to a default of 5.")
        )

    parser.add_argument(
        "--keys",
        type=str,
        nargs="*",
        default=None,
        help=("A choice of outputs from the local model to include in "
              "image encoding. The selected outputs are flattened and "
              "concatenated into a vector which represents the image. "
              "If no value is provided, we try to fall back on the "
              "choice in the 'model_info.json' file, and otherwise "
              "use the default of `['Eyx1', 'Varsyx1', 'Covyx1']`")
        )

    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help=("An optional folder in which to save the result of this "
              "experiment. Note that this is different from the folder "
              "in which the data is found. If no folder is given, the "
              "results are written into a temporary directory whose "
              "path can be found in `dataset.rootdir`.")
        )
    
    parser.add_argument(
        "--names",
        type=str,
        nargs="*",
        default=[],
        help=("A list of two names for the two categories, as in "
              "`--names SPOON FORK`. Will override any class names "
              "provided in the class names file in the data folder. "
              "A `ValueError` will be raised if not exact zero or "
              "two names are passed.")
        )
    
    parser.add_argument(
        "--extension",
        type=str,
        default=".avi",
        help=("The kind of video file to search for, as identified by "
              "its file name extension, such as '.avi' or '.mp4'.")
        )

    args, _ = parser.parse_known_args()
    
    model_class_name = None
    model_args = {}

    print("Received arguments\n")
    for key, value in args.__dict__.items():
        print("    %r: %r" % (key, value))
    print()

    # if model info is stored and you need it, load it from file:
    model_info_path = os.path.join(args.indir, "model_info.json")
    if os.path.isfile(model_info_path):
        with open(model_info_path, "rt") as source:
            model_args = json.load(source)
            print("Loaded model info %r\n" % model_args)
            model_class_name = model_args["class"]

    if type(args.model) == str and args.model:
        model_class_name = args.model

    if type(args.downsampling_factor) == int and args.downsampling_factor > 0:
        model_args["downsampling_factor"] = args.downsampling_factor

    if type(args.keys) == list and args.keys:
        model_args["keys"] = args.keys

    def filterargs(function, kwargs):
        """ Throw away dict elements that don't match function kwargs. """
        return {k: v for k, v in kwargs.items()
                if k in signature(function).parameters.keys()}

    # create image encoder:
    if model_class_name == "resnet":
        from encoding.image_encoding import ResNet50Encoder
        model_args = filterargs(ResNet50Encoder.__init__, model_args)
        encoder = ResNet50Encoder(**model_args)
    elif model_class_name == "local" or model_class_name == "OnnxModel":
        from encoding.pretrained import OnnxModel
        model_args = filterargs(OnnxModel.__init__, model_args)
        encoder = OnnxModel(**model_args)
    else:
        raise ValueError("Unrecognized model option %r" % (args.model,))
    
    # tell the user what you did (the `repr`` will also print the kwargs):
    print("Created image encoder %r\n" % encoder)

    # create a temporary outdir if one isn't given:
    if not args.outdir:
        tempdir = TemporaryDirectory()
        args.outdir = tempdir.name

    # initialize empty data set:
    dataset = EncodingData(encoder, rootdir=args.outdir)

    # get or potentially overwrite class names:
    if len(args.names) == 2:
        dataset.class_names[0] = args.names[0]
        dataset.class_names[1] = args.names[1]
    elif args.names:  # not length 0 (and not length 2)
        raise ValueError("Expected either a list of either 0 or 2 "
                         "class names, got %r" % (args.names,))
    else:
        try:
            dataset.load_class_names_from_file(args.indir)
        except FileNotFoundError:
            raise ValueError(
                "No class names provided through the `--names` argument, "
                "and no file of class names found in %" % (args.indir,)
                )

    # compile a dict of video paths with class indices:
    pathdict = compile_video_paths_and_index_dict(args.indir, args.extension)

    if not pathdict:
        print("Found no video files with extension %r in %r." %
              (args.extension, args.indir))
        sys.exit(0)

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
    print("Encoding images . . .")
    for vidpath, index in pathdict.items():
        # class_name = dataset.class_names[index]
        # print("Loading %r (class %r). . ." % (vidpath, class_name))
        dataset.initialize_recording(index)
        for frame in iter_video(vidpath):
            dataset.record_frame(frame)
        dataset.save_recording()
    print()

    # once you've collected all the codes, cross-validate:

    crossval_start = perf_counter()
    total_k = 0
    total_n = 0
    for class_index in range(2):
        class_name = dataset.class_names[class_index]
        print("CROSS-VALIDATING CLASS %r:\n" % class_name)
        sum_k = 0
        sum_n = 0
        iterator = dataset.crossval_accuracy(class_index)
        for eps, (k, n) in enumerate(iterator):
            print("Subset %r: %s/%s = %.1f pct" % (eps, k, n, 100.0 * k / n))
            sum_k += k
            sum_n += n
        print()
        print("Grand accuracy for class %r: %s/%s = %.1f pct\n" %
            (class_name, sum_k, sum_n, 100.0 * sum_k / sum_n))
        total_k += sum_k
        total_n += sum_n
    print("Grand accuracy for both classes: %s/%s = %.1f pct\n" %
          (total_k, total_n, 100.0 * total_k / total_n))
    crossval_dur = perf_counter() - crossval_start
    print("Cross-validation took a total of %.3f seconds.\n" % crossval_dur)