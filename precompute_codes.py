import os
import numpy as np
import cv2 as cv
from tqdm import tqdm
from argparse import ArgumentParser

from pretrained import encode


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
                        nargs="?",
                        help="file name extension to search for")

    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        raise ValueError("%r must be a folder")

    print("Seaching for %s files in %r . . ." % (args.filetype, args.folder,))
    filepaths_by_folder = {}
    for folderpath, _, filenames in os.walk(args.folder):
        filenames = [fn for fn in filenames if fn.endswith(args.filetype)]
        if filenames:
            foldername = os.path.basename(folderpath)
            filepaths = [os.path.join(folderpath, fn) for fn in filenames]
            filepaths_by_folder[foldername] = filepaths
    print("Found %s files:\n" % len(filepaths_by_folder))
    for dirname, filelist in filepaths_by_folder.items():
        print(dirname + "\n" + "\n".join(filelist) + "\n")

    for dirname, filelist in filepaths_by_folder.items():
        codesdict = {}
        for vidpath in filelist:
            firstname, _ = os.path.basename(vidpath).split(".")
            print("Computing codes for %s . . ." % vidpath)
            codes = [encode(normalize(f)) for f in iter_frames(vidpath)]
            print("Done.\n")
            codesdict[firstname] = np.array(codes)
        npz_path = os.path.join(args.folder, dirname + ".npz")
        np.savez(npz_path, **codesdict)
