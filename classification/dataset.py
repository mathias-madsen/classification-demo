import os
import cv2 as cv
from datetime import datetime
import json
import numpy as np

from gaussians.moments_tracker import MomentsTracker, combine
from gaussians import marginal_log_likelihoods as likes
from classification.discriminator import BiGaussianDiscriminator
from gaussians.marginal_log_likelihoods import pick_best_combination
from classification.cross_validation import random_splits


MAX_XVAL_STEPS_PER_CLASS = 5  # max num validation subsets per class


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

        print("Initializing recording for class %s" % class_index)
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

        print("Saving %s frames for class %s" %
              (self.current_tracker.count, self.currently_selected_class))

        self.video_writer.release()

        self.current_tracker.save(self.current_stats_path)
        np.save(self.current_codes_path, self.current_encoding_list)

        # import ipdb; ipdb.set_trace()
        episode_array = np.stack(self.current_encoding_list, axis=0)
        
        idx = self.currently_selected_class
        self.class_episode_codes[idx].append(episode_array)
        self.class_episode_stats[idx].append(self.current_tracker.copy())
        self.current_tracker.reset()

        print()
        for idx, trackerslist in self.class_episode_stats.items():
            if trackerslist:
                outpath = os.path.join(self.rootdir, "grand_stats_%s.npz" % idx)
                combined = combine(trackerslist)
                combined.save(outpath)
                print("Combined %s episodes for class %s into a single summary" %
                      (len(trackerslist), idx))
        print()

        self.currently_selected_class = None
        del self.current_image_list

    def discard_recording(self):
        self.video_writer.release()
        os.remove(self.current_video_path)

    def num_examples_per_class(self):
        return {self.class_names[idx]: sum(len(e) for e in collection)
                for idx, collection in self.class_episode_codes.items()}

    def all_classes_nonempty(self):
        return all(n >= 1 for n in self.num_examples_per_class().values())

    def crossval_logp(self, class_index):

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
    
    def compute_num_cross_val_steps(self):
        lengths = [len(v) for v in self.class_episode_stats.values()]
        capped_lengths = [min(n, MAX_XVAL_STEPS_PER_CLASS) for n in lengths]
        return sum(n_eps if n_eps >= 2 else 0 for n_eps in capped_lengths)

    def crossval_accuracy_0(self):
        len0, len1 = [len(v) for v in self.class_episode_codes.values()]
        if len0 < 2 or len1 < 1:
            return
        discriminator = BiGaussianDiscriminator()
        codes_0 = self.class_episode_codes[0]
        for test_idx in random_splits(len(codes_0), MAX_XVAL_STEPS_PER_CLASS):
            train_idx = [i for i in range(len(codes_0)) if i not in test_idx]
            test_codes = np.concatenate([codes_0[i] for i in test_idx], axis=0)
            stats_0 = self.class_episode_stats[0].copy()
            stats_0 = [stats_0[i] for i in train_idx]
            stats_1 = self.class_episode_stats[1]
            stats = pick_best_combination(combine(stats_0), combine(stats_1))
            discriminator.set_stats(*stats)
            corrects = discriminator.classify(test_codes) == 1
            yield sum(corrects), len(corrects)

    def crossval_accuracy_1(self):
        len0, len1 = [len(v) for v in self.class_episode_codes.values()]
        if len0 < 1 or len1 < 2:
            return
        discriminator = BiGaussianDiscriminator()
        codes_1 = self.class_episode_codes[1]
        for test_idx in random_splits(len(codes_1), MAX_XVAL_STEPS_PER_CLASS):
            train_idx = [i for i in range(len(codes_1)) if i not in test_idx]
            test_codes = np.concatenate([codes_1[i] for i in test_idx], axis=0)
            stats_0 = self.class_episode_stats[0]
            stats_1 = self.class_episode_stats[1].copy()
            stats_1 = [stats_1[i] for i in train_idx]
            stats = pick_best_combination(combine(stats_0), combine(stats_1))
            discriminator.set_stats(*stats)
            corrects = discriminator.classify(test_codes) == 2
            yield sum(corrects), len(corrects)
    
    def crossval_accuracy(self, class_index):
        if class_index == 0:
            return self.crossval_accuracy_0()
        elif class_index == 1:
            return self.crossval_accuracy_1()
        else:
            raise ValueError("Unexpected class index %r" % class_index)
