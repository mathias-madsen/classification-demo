"""
This script is for offline testing of the classification method.

It takes one argument, which is a path to a folder containing two
subfolders, 
"""


import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy import ndimage
from argparse import ArgumentParser

# from image_encoding import encode
from pretrained import encode
from discriminator import BiGaussianDiscriminator


def collect_npz_file_paths(folder):
    """ Compile a {subfolder: npzpath} dict. """
    print("Seaching for .npz files in %r . . ." % (args.folder,))
    filedict = {}
    for root, _, filenames in os.walk(folder):
        for fn in filenames:
            if fn.endswith(".npz"):
                fp = os.path.join(root, fn)
                dirname = os.path.basename(root)
                if dirname in filedict:
                        filedict[dirname].append(fp)
                else:
                        filedict[dirname] = [fp]
    print("Found the following classes:")
    for dirname, pathlist in filedict.items():
         print("%r: %s videos" % (dirname, len(pathlist)))
    print()
    return filedict


def load_video(path):
    """ Load a single video from NPZ, and print some info. """
    print(path, ". . .", end=" ")
    with np.load(path) as source:
        video = source["arr_0"]
    print("Video shape: %s" % (video.shape,))
    return video


def vid2codes(video):
    """ Compute the codes of the images in a video snippet. """
    return np.stack([encode(f) for f in tqdm(video)], axis=0)


def split(episodes):
    " Cut off one episode and concatenate the rest. "
    test_id = np.random.randint(len(episodes))
    train_eps = [e for i, e in enumerate(episodes) if i != test_id]
    test_eps = [e for i, e in enumerate(episodes) if i == test_id]
    train_frames = np.concatenate(train_eps, axis=0)
    test_frames = np.concatenate(test_eps, axis=0)
    return train_frames, test_frames
    

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("folder",
                        type=str,
                        help="folder in which to look for videos")
    parser.add_argument("--smoothing",
                        type=int,
                        help="number of frames to take into account for a decision",
                        default=5)
    args = parser.parse_args()
    if not os.path.isdir(args.folder):
        raise ValueError("%r must be a folder")

    filedict = collect_npz_file_paths(args.folder)

    print("Loading videos . . .")
    videodict = {subfolder: [load_video(p) for p in pathlist]
                 for subfolder, pathlist in filedict.items()}
    print()
                
    print("Computing codes . . .")
    codesdict = {folder: [vid2codes(v) for v in videolist]
                 for folder, videolist in videodict.items()}
    print("Done.\n")

    class_a_name, class_b_name = filedict.keys()
    discriminator = BiGaussianDiscriminator()

    for _ in range(5):

        train_a, test_a = split(codesdict[class_a_name])
        train_b, test_b = split(codesdict[class_b_name])    
        discriminator.fit(train_a, train_b)

        framewise_scores_a = discriminator(test_a)
        framewise_scores_b = discriminator(test_b)

        kernel = np.ones(args.smoothing) / args.smoothing
        scores_a = np.convolve(framewise_scores_a, kernel)
        scores_b = np.convolve(framewise_scores_b, kernel)

        print("Test results:")
        print("|         | %r | %r |" % (class_a_name, class_b_name))
        print("| :-----: | :-----: | :-----: |")
        print("| %s | %s | %s |" % (class_a_name,
                                    np.sum(scores_a >= 0),
                                    np.sum(scores_a < 0)))
        print("| %s | %s | %s |" % (class_b_name,
                                    np.sum(scores_b >= 0),
                                    np.sum(scores_b < 0)))
        print()

    figure, axes = plt.subplots(figsize=(12, 8))
    axes.hist(scores_a, bins=51, alpha=0.5, label=class_a_name)
    axes.hist(scores_b, bins=51, alpha=0.5, label=class_b_name)
    plt.suptitle("Test scores")
    plt.legend()
    plt.tight_layout()
    plt.show()

    colors = np.random.uniform(0, 1, size=(len(codesdict), 3))
    nrows = sum(len(v) for v in codesdict.values())
    figure, axlist = plt.subplots(nrows=nrows, sharey=True, figsize=(12, 8))
    i = 0
    for (class_name, codes_list), color in zip(codesdict.items(), colors):
        for codes in codes_list:
            evidence = discriminator(codes)
            axlist[i].plot(evidence, ".", color=color, alpha=0.5)
            axlist[i].plot(0 * evidence, "-", color="red", alpha=0.8)
            axlist[i].set_title(class_name)
            i += 1
    figure.tight_layout()
    separation_figure_path = os.path.join(args.folder, "separation.png")
    figure.savefig(separation_figure_path)
    plt.show()

    frame = next(iter(videodict.values()))[0][0]
    figure, axes = plt.subplots(figsize=(8, 7))
    panel = axes.imshow(frame)
    axes.axis("off")
    axes.set_title("XXX", fontsize=24)
    figure.tight_layout()
    plt.pause(0.01)
    for name in videodict.keys():
        for codelist, videolist in zip(codesdict[name], videodict[name]):
            evidence_history = np.array([discriminator(c) for c in codelist])
            evidence_history = np.convolve(evidence_history, kernel)
            for frame, evidence in zip(videolist[::5], evidence_history[::5]):
                panel.set_data(frame)
                color = "black" if evidence < 0 else "blue"
                axes.set_title("%.5f" % evidence, fontsize=24, color=color)
                plt.pause(1 / 15)
                if not plt.fignum_exists(figure.number):
                    break
            if not plt.fignum_exists(figure.number):
                break
