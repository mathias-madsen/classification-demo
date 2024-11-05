import os
import cv2 as cv
from datetime import datetime
import json
import numpy as np

from gaussians.moments_tracker import MomentsTracker, combine
from gaussians import marginal_log_likelihoods as likes


def invent_name():
    """ Return a string that is part time information, part noise. """
    head = datetime.now().strftime("%Y%m%d-%H%M%S")
    alphabet = [chr(i) for i in range(65, 91)]
    tail = "".join(np.random.choice(alphabet, size=6))
    return head + "-" + tail


class EncodingData:
    """ An updatable set of examples for two classes, with a fitted model. """

    def __init__(self, image_encoder, rootdir):

        self.class_episode_codes = {0: [], 1: []}
        self.class_episode_stats = {0: [], 1: []}
        self.class_names = {0: "", 1: ""}
        self.currently_selected_class = None

        self.rootdir = rootdir
        for idx in self.class_episode_stats.keys():
            os.makedirs(os.path.join(self.rootdir, str(idx)), exist_ok=True)
        print("Saving artifacts to %r.\n" % self.rootdir)
        
        self.image_encoder = image_encoder
        self.save_model_information()

        self.current_video_path = ""
        self.current_stats_path = ""
        self.current_codes_path = ""

        self.image_height = None
        self.image_width = None
        self.current_tracker = None

    def get_current_class_name(self):
        return self.class_names[self.currently_selected_class]

    def load_class_names_from_file(self, path):
        if not path.endswith(".csv"):
            path = os.path.join(path, "class_indices_and_names.csv")
        with open(path, "rt") as source:
            for line in source.read().strip().split("\n"):
                idx_string, name_string = line.split(",")
                self.class_names[int(idx_string)] = name_string

    def compute_dimensions(self, dummy_image):

        dummy_encoding = self.image_encoder(dummy_image)
        dim, = dummy_encoding.shape
        print("Latent dim: %s" % (dim,))
        self.image_height, self.image_width, _ = dummy_image.shape
        print("Image shape: %sx%s" % (self.image_height, self.image_width))
        print("")

        self.current_tracker = MomentsTracker(np.zeros(dim), np.eye(dim), 0)

    def save_model_information(self):
        jpath = os.path.join(self.rootdir, "model_info.json")
        jdata = {
            "class": self.image_encoder.__class__.__name__,
            "downsampling_factor": self.image_encoder.downsampling_factor,
            }
        if hasattr(self.image_encoder, "keys"):
            jdata["keys"] = self.image_encoder.keys
        with open(jpath, "wt") as target:
            json.dump(jdata, target, indent=4)
        print("Saved model information to %r.\n" % jpath)

    def save_class_names_to_file(self):
        outpath = os.path.join(self.rootdir, "class_indices_and_names.csv")
        with open(outpath, "wt") as target:
            for (k, v) in self.class_names.items():
                target.write("%s,%s\n" % (k, v))

    def initialize_recording(self, class_index):

        self.currently_selected_class = class_index

        self.current_image_list = []
        self.current_encoding_list = []
        self.current_tracker.reset()

        name = invent_name()
        folder = os.path.join(self.rootdir, str(class_index))
        self.current_video_path = os.path.join(folder, name + ".avi")
        self.current_stats_path = os.path.join(folder, name + ".npz")
        self.current_codes_path = os.path.join(folder, name + ".npy")

        self.video_writer = cv.VideoWriter(
            filename=self.current_video_path,
            fourcc=cv.VideoWriter_fourcc(*"MJPG"),
            fps=16,
            frameSize=(self.image_width, self.image_height),
            )
        
    def record_frame(self, rgb):

        self.current_image_list.append(rgb)
        self.video_writer.write(rgb[:, :, ::-1])  # write BGR to .avi

        latent_vector = self.image_encoder(rgb)
        self.current_tracker.update_with_single(latent_vector)
        self.current_encoding_list.append(latent_vector)

    def save_recording(self):

        self.video_writer.release()

        self.current_tracker.save(self.current_stats_path)
        np.save(self.current_codes_path, self.current_encoding_list)

        # import ipdb; ipdb.set_trace()
        episode_array = np.stack(self.current_encoding_list, axis=0)
        
        idx = self.currently_selected_class
        self.class_episode_codes[idx].append(episode_array)
        self.class_episode_stats[idx].append(self.current_tracker.copy())
        self.current_tracker.reset()

        for idx, trackerslist in self.class_episode_stats.items():
            if trackerslist:
                outpath = os.path.join(self.rootdir, "grand_stats_%s.npz" % idx)
                combined = combine(trackerslist)
                combined.save(outpath)

        self.currently_selected_class = None
        del self.current_image_list

    def cross_validate_stats(self, class_index):

        stats_list = self.class_episode_stats[class_index]
        name = self.class_names[class_index]
        
        if not stats_list:
            print("Cannot cross-validate empty stats list "
                  "(index %s, name %r)" % (class_index, name))
            return None, None

        if len(stats_list) == 1:
            print("Cannot cross-validate singleton stats list "
                  "(index %s, name %r)" % (class_index, name))
            return None, None

        dim, = stats_list[0].mean.shape
        prior = MomentsTracker(np.zeros(dim), np.eye(dim), 1.0)
        all_uncorrelated = []
        all_correlated = []
        for i, test_stats in enumerate(stats_list):
            train_stats = stats_list.copy()
            train_stats.pop(i)
            train_stats = combine(train_stats)
            train_stats = combine([train_stats, prior])
            df = train_stats.count + 1.0
            uncr = likes.uncorrelated_loglikes(*test_stats, *train_stats, df)
            all_uncorrelated.append(uncr)
            corr = likes.correlated_loglikes(*test_stats, *train_stats, df)
            all_correlated.append(corr)
        
        return sum(all_correlated), sum(all_uncorrelated)

    def discard_recording(self):
        self.video_writer.release()
        os.remove(self.current_video_path)

    def num_examples_per_class(self):
        return {self.class_names[idx]: sum(len(e) for e in collection)
                for idx, collection in self.class_episode_codes.items()}

    def all_classes_nonempty(self):
        return all(n >= 1 for n in self.num_examples_per_class().values())


if __name__ == "__main__":

    from pretrained import OnnxModel
    from scipy.datasets import face
    from tqdm import tqdm
    import shutil

    raccoon = face()

    small_height = 256
    small_width = 320
    big_height, big_width, _ = raccoon.shape
    margin = min(big_height - small_height, big_width - small_width)

    def get_raccoon_from_index(index):
        return raccoon[index:index + small_height, index:index + small_width]

    def get_raccoonn_from_time(time, period=10):
        unit = (1 + np.sin(time / period)) / 2
        index = int(round(margin * unit))
        detail = get_raccoon_from_index(index)
        return detail

    encoder = OnnxModel()

    folder = "/tmp/test_delete_me/"
    shutil.rmtree(folder)
    os.makedirs(folder)

    dummy_image = get_raccoon_from_index(0)
    dataset = EncodingData(encoder, folder)

    dataset.compute_dimensions(dummy_image)
    
    dataset.class_names[0] = "LEFT"
    dataset.class_names[1] = "RIGHT"

    dataset.save_class_names_to_file()
    dataset.save_model_information()

    dataset.initialize_recording(class_index=0)
    for t in tqdm(range(20)):
        dataset.record_frame(get_raccoonn_from_time(t))
    dataset.save_recording()

    dataset.initialize_recording(class_index=0)
    for t in tqdm(range(30)):
        dataset.record_frame(get_raccoonn_from_time(t))
    dataset.save_recording()

    dataset.initialize_recording(class_index=1)
    for t in tqdm(range(40)):
        dataset.record_frame(get_raccoonn_from_time(t))
    dataset.save_recording()

    dataset.initialize_recording(class_index=1)
    for t in tqdm(range(20)):
        dataset.record_frame(get_raccoonn_from_time(t))
    dataset.discard_recording()


