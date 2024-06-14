import os
import numpy as np
import cv2 as cv
from tqdm import tqdm
from argparse import ArgumentParser


def normalize(bgr):

    # convert from blue-green-red to red-green-blue:
    rgb = bgr[:, :, ::-1]

    # convert to landscape format if necessary:
    height, width, _ = rgb.shape
    if height > width:
        rgb = np.transpose(rgb, [1, 0, 2])

    # adjust aspect ratio if necessary:
    height, width, _ = rgb.shape
    if width / height > 5/4:
        excess = width - int(height * 5/4)
        left = excess // 2
        right = excess - left
        rgb = rgb[:, left:-right, :]
    
    return rgb


def iter_frames(path):
        capture = cv.VideoCapture(path)
        length = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
        for _ in tqdm(range(length)):
            success, frame = capture.read()
            assert success
            yield frame


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("folder",
                        type=str,
                        help="folder in which to look for videos")

    parser.add_argument("filetype",
                        type=str,
                        default=".mp4",
                        help="file name extension to search for")

    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        raise ValueError("%r must be a folder")

    print("Seaching for %s files in %r . . ." % (args.filetype, args.folder,))
    inout = {}
    for root, _, filenames in os.walk(args.folder):
        for fn in filenames:
            inpath = os.path.join(root, fn)
            if inpath.endswith(args.filetype):
                outpath = inpath[:-4] + ".npz"
                inout[inpath] = outpath
                # if not os.path.isfile(outpath):
                #     inout[inpath] = outpath
    print("Found %s files:" % len(inout))
    for inpath, outpath in inout.items():
        print(inpath, "=>", outpath)
    print()

    for inpath, outpath in inout.items():
        print("Loading from %s . . ." % args.filetype)
        frames = [normalize(f) for f in iter_frames(inpath)]
        print("Done.\n")

        print("Saving as .npz . . .")
        np.savez(outpath, np.stack(frames, axis=0))
        print("Done.\n")
