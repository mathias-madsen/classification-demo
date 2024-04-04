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

from image_encoding import encode
from discriminator import BiGaussianDiscriminator



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("folder",
                        type=str,
                        help="folder in which to look for videos")
    args = parser.parse_args()
    if not os.path.isdir(args.folder):
        raise ValueError("%r must be a folder")

    print("Seaching for .npz files in %r . . ." % (args.folder,))
    filedict = {}
    for root, _, filenames in os.walk(args.folder):
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

    print("Loading videos . . .")
    videodict = {}
    for dirname, pathlist in filedict.items():
        videodict[dirname] = []
        for path in pathlist:
            print(path, ". . .", end=" ")
            with np.load(path) as source:
                video = source["arr_0"]
                videodict[dirname].append(video)
            print("Video shape: %s" % (video.shape,))
    print()
                
    print("Computing codes . . .")
    codesdict = {}
    for dirname, videolist in videodict.items():
         codesdict[dirname] = []
         for video in videolist:
              codes = np.stack([encode(f) for f in tqdm(video)], axis=0)
              codesdict[dirname].append(codes)
    print("Done.\n")

    def split(episodes):
         " Cut off one episode and concatenate the rest. "
         test_id = np.random.randint(len(episodes))
         train_eps = [e for i, e in enumerate(episodes) if i != test_id]
         test_eps = [e for i, e in enumerate(episodes) if i == test_id]
         train_frames = np.concatenate(train_eps, axis=0)
         test_frames = np.concatenate(test_eps, axis=0)
         return train_frames, test_frames
    
    class_a_name, class_b_name = filedict.keys()
    train_a, test_a = split(codesdict[class_a_name])
    train_b, test_b = split(codesdict[class_b_name])
    
    discriminator = BiGaussianDiscriminator()
    discriminator.fit(train_a, train_b)

    scores_a = discriminator(test_a)
    scores_b = discriminator(test_b)

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
    plt.show()

    frame = next(iter(videodict.values()))[0][0]
    figure, axes = plt.subplots(figsize=(8, 7))
    panel = axes.imshow(frame)
    axes.axis("off")
    axes.set_title("XXX", fontsize=24)
    figure.tight_layout()
    for name in videodict.keys():
        for codelist, videolist in zip(codesdict[name], videodict[name]):
            for code, frame in zip(codelist[::5], videolist[::10]):
                panel.set_data(frame)
                evidence = discriminator(code)
                color = "black" if evidence < 0 else "blue"
                axes.set_title("%.1f" % evidence, fontsize=24, color=color)
                plt.pause(1 / 15)
                if not plt.fignum_exists(figure.number):
                    break
            if not plt.fignum_exists(figure.number):
                break
